"""Small, serializable residual models for aligned component predictions.

These heads do not read MRI images. Their inputs must be probabilities produced
by the same four component families used during training, in the saved order.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


METHODS = {
    "convex_probability": {"strength": 0.01, "max_iterations": 500},
    "logit_same_class": {"strength": 0.02, "intercept_strength": 0.01, "max_iterations": 500},
    "logit_cross_class": {"strength": 0.08, "intercept_strength": 0.02, "max_iterations": 500},
}
EPSILON = 1e-6


def validate_probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4, 12) or not len(p):
        raise ValueError("Expected a nonempty [studies, four components, twelve targets] array")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Component probabilities must be finite and in [0,1]")
    return p


def logits(values):
    p = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(p) - np.log1p(-p)


def target_features(logit_values, target, method):
    if method == "logit_same_class":
        return logit_values[:, :, target]
    if method == "logit_cross_class":
        return logit_values.reshape(len(logit_values), -1)
    raise ValueError("Not a residual-logit method: " + method)


def fit_combiner(probabilities, labels, confidence, method):
    """Fit one fixed model per target, using only caller-provided training rows."""
    p = validate_probabilities(probabilities)
    y, weights = np.asarray(labels, dtype=np.float64), np.asarray(confidence, dtype=np.float64)
    if y.shape != (len(p), 12) or weights.shape != y.shape:
        raise ValueError("Labels and confidence must match studies and twelve targets")
    if method not in METHODS:
        raise ValueError("Unknown fixed method: " + method)
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("Weak-label probabilities must be finite and in [0,1]")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Confidence weights must be finite and nonnegative")
    config = METHODS[method]
    lp = logits(p)
    heads = []
    for target in range(12):
        valid = (weights[:, target] > 0) & (np.abs(y[:, target] - 0.5) > EPSILON)
        if valid.sum() < 5 or len(np.unique(y[valid, target] > 0.5)) != 2:
            raise ValueError("Insufficient unambiguous training labels for target " + str(target))
        yy = y[valid, target]
        ww = weights[valid, target]
        ww = ww / ww.sum()
        if method == "convex_probability":
            xx = p[valid, :, target]
            prior = np.array([1.0, 0.0, 0.0, 0.0])

            def objective(coef):
                z = np.clip(xx @ coef, EPSILON, 1 - EPSILON)
                loss = -np.sum(ww * (yy * np.log(z) + (1 - yy) * np.log1p(-z)))
                delta = coef - prior
                gradient = xx.T @ (ww * (z - yy) / (z * (1 - z)))
                return loss + 0.5 * config["strength"] * (delta @ delta), gradient + config["strength"] * delta

            solution = minimize(objective, prior, jac=True, method="SLSQP", bounds=[(0, 1)] * 4,
                                constraints=[{"type": "eq", "fun": lambda c: c.sum() - 1,
                                              "jac": lambda c: np.ones_like(c)}],
                                options={"maxiter": config["max_iterations"], "ftol": 1e-10})
            if not solution.success:
                raise RuntimeError("Convex optimizer failed: " + str(solution.message))
            coef = np.maximum(solution.x, 0)
            coef /= coef.sum()
            head = {"weights": coef.tolist()}
        else:
            raw = target_features(lp, target, method)[valid]
            mean, std = raw.mean(0), raw.std(0)
            std = np.where(std < 1e-6, 1, std)
            xx = (raw - mean) / std
            offset = lp[valid, 0, target]

            def objective(theta):
                coef, intercept = theta[:-1], theta[-1]
                z = offset + xx @ coef + intercept
                delta = ww * (expit(z) - yy)
                loss = np.sum(ww * (np.logaddexp(0, z) - yy * z))
                penalty = 0.5 * config["strength"] * (coef @ coef) + 0.5 * config["intercept_strength"] * intercept ** 2
                gradient = np.r_[xx.T @ delta + config["strength"] * coef,
                                 delta.sum() + config["intercept_strength"] * intercept]
                return loss + penalty, gradient

            solution = minimize(objective, np.zeros(xx.shape[1] + 1), jac=True, method="L-BFGS-B",
                                options={"maxiter": config["max_iterations"], "ftol": 1e-11, "gtol": 1e-7})
            if not solution.success:
                raise RuntimeError("Residual optimizer failed: " + str(solution.message))
            head = {"coefficients": solution.x[:-1].tolist(), "intercept": float(solution.x[-1]),
                    "feature_mean": mean.tolist(), "feature_std": std.tolist()}
        head.update({"training_rows": int(valid.sum()), "optimizer_success": bool(solution.success),
                     "optimizer_iterations": int(solution.nit), "training_objective": float(solution.fun)})
        heads.append(head)
    return {"schema": "rsna_probability_residual_v1", "method": method, "configuration": config.copy(),
            "input_shape": ["studies", 4, 12], "MIL_component_index": 0,
            "cross_class_feature_order": "component-major, target-minor", "heads": heads}


def predict_combiner(model, probabilities):
    p = validate_probabilities(probabilities)
    method = model["method"]
    if method not in METHODS or len(model["heads"]) != 12:
        raise ValueError("Unknown or incomplete combiner model")
    lp = logits(p)
    output = np.empty((len(p), 12), np.float64)
    for target, head in enumerate(model["heads"]):
        if method == "convex_probability":
            weights = np.asarray(head["weights"], dtype=np.float64)
            if weights.shape != (4,) or not np.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum()-1) > 1e-8:
                raise ValueError("Invalid convex weights")
            output[:, target] = p[:, :, target] @ weights
        else:
            raw = target_features(lp, target, method)
            mean, std = np.asarray(head["feature_mean"]), np.asarray(head["feature_std"])
            coef = np.asarray(head["coefficients"])
            if mean.shape != (raw.shape[1],) or std.shape != mean.shape or coef.shape != mean.shape:
                raise ValueError("Residual feature dimensions do not match the checkpoint")
            if not np.isfinite(np.r_[mean, std, coef, head["intercept"]]).all() or (std <= 0).any():
                raise ValueError("Invalid residual checkpoint")
            output[:, target] = expit(lp[:, 0, target] + ((raw-mean)/std) @ coef + head["intercept"])
    if not np.isfinite(output).all() or (output < 0).any() or (output > 1).any():
        raise RuntimeError("Combiner produced invalid probabilities")
    return output
