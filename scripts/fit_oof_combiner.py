#!/usr/bin/env python3
"""Fit original CPU residual heads using explicit, external prior OOF inputs.

No study identifiers, report hashes, labels, predictions, or clinical data are
written into the project. Only protocol, aggregate metrics, and learned numeric
model parameters are exported. Reused base folds make this exploratory CV.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna.combiner import METHODS, fit_combiner, predict_combiner, validate_probabilities

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMPONENTS = ["prior_convnext_MIL", "linear_pool", "sequence_transformer", "texture_boost"]
UID = "StudyInstanceUID"
LIMITATIONS = [
    "Weak report labels are not expert-label competition truth.",
    "Base OOF folds were reused; the prior MIL selected its best validation epoch.",
    "Base learners in other folds were trained using groups held out from a second-level fit. This is not nested independent cross-validation.",
    "All prior folds and component results had already been examined. There is no newly untouched holdout.",
    "No matched OOF predictions from the complete scored 0.943 or Public950 ensemble are available. No full-ensemble improvement is established.",
    "Intervals condition on fixed OOF predictions and do not include model-retraining or selection uncertainty.",
    "The learned heads require the exact four old component families at inference; they cannot be plugged into Public950 without those inputs.",
    "No expert Gold study or identical-report relative is used for fitting, evaluation, or model selection.",
]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def fold_ranks(predictions, folds):
    p = np.asarray(predictions, np.float64)
    out = np.empty_like(p)
    for fold in sorted(set(folds.tolist())):
        selected = folds == fold
        out[selected] = rankdata(p[selected], axis=0, method="average") / selected.sum()
    return out


def make_auc_cache(labels, predictions):
    cache = []
    for target in range(12):
        eligible = np.isfinite(labels[:, target]) & (np.abs(labels[:, target]-0.5) > 1e-6)
        yy = labels[eligible, target] > 0.5
        _, score_group = np.unique(predictions[eligible, target], return_inverse=True)
        cache.append((eligible, yy, score_group, int(score_group.max())+1))
    return cache


def auc_from_cache(cache, counts):
    values = []
    for eligible, yy, groups, n_groups in cache:
        ww = counts[eligible]
        positives = np.bincount(groups, weights=ww*yy, minlength=n_groups)
        negatives = np.bincount(groups, weights=ww*(~yy), minlength=n_groups)
        denominator = positives.sum()*negatives.sum()
        if denominator <= 0:
            return None
        credit = np.sum(positives*(np.cumsum(negatives)-0.5*negatives))
        values.append(float(credit/denominator))
    return np.array(values)


def score(labels, predictions):
    values = auc_from_cache(make_auc_cache(labels, predictions), np.ones(len(labels)))
    if values is None:
        raise ValueError("One target lacks both classes")
    return {"macro_auc": float(values.mean()), "per_target_auc": dict(zip(TARGETS, map(float, values)))}


def read_external_data(args):
    with np.load(args.targets, allow_pickle=False) as arrays:
        ids_all = arrays["ids"].astype(str)
        labels_all = arrays["y"].astype(np.float64)
        confidence_all = arrays["weights"].astype(np.float64)
        gold = arrays["gold"].astype(bool)
        eligible = arrays["eligible"].astype(bool)
        folds_all = arrays["folds"].astype(int)
    assert len(set(ids_all.tolist())) == len(ids_all), "Duplicate target UIDs"
    assert not (gold & eligible).any(), "Gold cannot be eligible"
    ids = ids_all[eligible]
    labels, confidence, folds = labels_all[eligible], confidence_all[eligible], folds_all[eligible]
    assert np.isfinite(labels).all() and np.isfinite(confidence).all()
    manifest = pd.read_csv(args.manifest, dtype={UID: str, "report_sha256": str})
    assert not manifest[UID].duplicated().any(), "Duplicate manifest UIDs"
    manifest = manifest.set_index(UID).loc[ids]
    assert not manifest["is_gold"].any() and not manifest["gold_report_duplicate"].any()
    assert manifest["usable_pixels"].all()
    assert np.array_equal(manifest["fold"].to_numpy(int), folds)
    assert (manifest.groupby("report_sha256")["fold"].nunique() == 1).all(), "Report groups cross folds"
    components = []
    for path in [args.mil, args.linear, args.sequence, args.texture]:
        frame = pd.read_csv(path, dtype={UID: str})
        assert not frame[UID].duplicated().any(), "Duplicate prediction UIDs"
        assert set(frame[UID]) == set(ids), "Prediction cohort mismatch"
        frame = frame.set_index(UID).loc[ids]
        assert np.array_equal(frame["fold"].to_numpy(int), folds), "Prediction fold mismatch"
        components.append(frame[TARGETS].to_numpy(np.float64))
    probabilities = validate_probabilities(np.stack(components, axis=1))
    groups, _ = pd.factorize(manifest["report_sha256"], sort=True)
    return probabilities, labels, confidence, folds, groups, int(gold.sum()), int((~eligible).sum())


def bootstrap(labels, ranked_predictions, folds, groups, repetitions, seed):
    caches = {name: make_auc_cache(labels, pred) for name, pred in ranked_predictions.items()}
    group_count = int(groups.max()) + 1
    fold_groups = {int(fold): np.unique(groups[folds == fold]) for fold in np.unique(folds)}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in caches}
    for _ in range(repetitions):
        group_weights = np.zeros(group_count, dtype=np.int64)
        for gg in fold_groups.values():
            sampled = rng.choice(gg, size=len(gg), replace=True)
            group_weights += np.bincount(sampled, minlength=group_count)
        counts = group_weights[groups]
        values = {name: auc_from_cache(cache, counts) for name, cache in caches.items()}
        if any(x is None for x in values.values()):
            continue
        for name, value in values.items():
            draws[name].append(value)
    matrix = {name: np.stack(value) for name, value in draws.items()}
    incumbent = matrix["MIL"]
    output = {}
    for name in caches:
        delta = matrix[name]-incumbent
        macro = delta.mean(axis=1)
        output[name] = {
            "macro_paired_95percent_interval_vs_MIL": np.quantile(macro, [0.025, 0.975]).tolist(),
            "per_target_paired_95percent_interval_vs_MIL": {
                target: np.quantile(delta[:, j], [0.025, 0.975]).tolist() for j, target in enumerate(TARGETS)},
        }
        if name in METHODS:
            # Three fixed candidate models, not twelve individually tuned targets.
            output[name]["macro_bonferroni_three_candidate_95percent_interval_vs_MIL"] = np.quantile(macro, [0.025/3, 1-0.025/3]).tolist()
    return {"seed": seed, "requested_repetitions": repetitions, "valid_repetitions": len(incumbent),
            "sampling": "paired report groups, stratified within the five original folds",
            "contrasts": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("targets", "manifest", "mil", "linear", "sequence", "texture"):
        parser.add_argument("--"+option, type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    assert args.bootstrap_repetitions >= 100
    input_paths = {name: getattr(args, name) for name in ("targets", "manifest", "mil", "linear", "sequence", "texture")}
    protocol = {
        "schema": "rsna_original_oof_combiner_protocol_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS, "component_order": COMPONENTS, "target_order": TARGETS,
        "training_loss": "Confidence-weighted soft-label cross entropy; exact-0.5 labels excluded",
        "validation_truth": "Weak-label probability above 0.5; exact-0.5 cells excluded",
        "primary_model_selection": "Highest five-fold within-fold-percentile-normalized OOF macro AUC among the three fixed heads; alphabetical method name breaks exact ties",
        "baselines": ["MIL raw and fold-ranked", "fixed 90% MIL rank + 10% texture rank, within original folds"],
        "bootstrap_repetitions": args.bootstrap_repetitions, "bootstrap_seed": 20261009,
        "limitations": LIMITATIONS,
        "no_automatic_production_replacement": True,
        "input_sha256": {name: sha_file(path) for name, path in input_paths.items()},
    }
    if args.protocol.exists():
        existing = json.loads(args.protocol.read_text())
        for key in ("methods", "component_order", "target_order", "primary_model_selection", "bootstrap_repetitions", "input_sha256"):
            assert existing[key] == protocol[key], "Protocol changed: " + key
        protocol = existing
    else:
        # Freeze before reading labels, fitting, or looking at any new result.
        save_json(args.protocol, protocol)
    start = time.monotonic()
    p, y, confidence, folds, groups, gold_count, excluded_count = read_external_data(args)
    fold_counts = {str(int(k)): int((folds == k).sum()) for k in np.unique(folds)}
    output = {"MIL": p[:, 0, :],
              "MIL_texture_fixed_90_10": 0.9*fold_ranks(p[:, 0, :], folds)+0.1*fold_ranks(p[:, 3, :], folds)}
    fold_fit_summaries = {}
    for method in sorted(METHODS):
        out = np.full_like(y, np.nan)
        summaries = []
        for fold in sorted(np.unique(folds)):
            train, valid = folds != fold, folds == fold
            assert not set(groups[train]) & set(groups[valid]), "Report leakage in meta fold"
            model = fit_combiner(p[train], y[train], confidence[train], method)
            out[valid] = predict_combiner(model, p[valid])
            summaries.append({"fold": int(fold), "train_studies": int(train.sum()), "validation_studies": int(valid.sum()),
                              "optimizer_success_all_targets": all(h["optimizer_success"] for h in model["heads"]),
                              "optimizer_iterations_per_target": {t: h["optimizer_iterations"] for t, h in zip(TARGETS, model["heads"])},
                              "probability_weights": {t: h["weights"] for t, h in zip(TARGETS, model["heads"])} if method == "convex_probability" else None})
            print(f"Completed {method} fold {fold}", flush=True)
        assert np.isfinite(out).all()
        output[method] = out
        fold_fit_summaries[method] = summaries
    ranked = {name: fold_ranks(pred, folds) for name, pred in output.items()}
    metrics = {}
    for name, pred in output.items():
        metrics[name] = {"raw_pooled_oof": score(y, pred), "fold_ranked_oof": score(y, ranked[name]),
                         "per_fold": {str(int(k)): score(y[folds==k], pred[folds==k]) for k in np.unique(folds)}}
    base = metrics["MIL"]["fold_ranked_oof"]
    for name, metric in metrics.items():
        current = metric["fold_ranked_oof"]
        metric["fold_ranked_delta_vs_MIL"] = {
            "macro_auc": current["macro_auc"]-base["macro_auc"],
            "per_target_auc": {t: current["per_target_auc"][t]-base["per_target_auc"][t] for t in TARGETS}}
        metric["per_fold_delta_vs_MIL"] = {
            fold: {"macro_auc": values["macro_auc"]-metrics["MIL"]["per_fold"][fold]["macro_auc"],
                   "per_target_auc": {t: values["per_target_auc"][t]-metrics["MIL"]["per_fold"][fold]["per_target_auc"][t] for t in TARGETS}}
            for fold, values in metric["per_fold"].items()}
    selected = sorted(METHODS, key=lambda name: (-metrics[name]["fold_ranked_oof"]["macro_auc"], name))[0]
    bootstrap_result = bootstrap(y, ranked, folds, groups, args.bootstrap_repetitions, 20261009)
    final_models = {}
    for method in sorted(METHODS):
        model = fit_combiner(p, y, confidence, method)
        final_prediction = predict_combiner(model, p)
        assert np.isfinite(final_prediction).all()
        model.update({"component_order": COMPONENTS, "target_order": TARGETS})
        final_models[method] = model
    models = {"schema": "rsna_original_oof_combiner_bundle_v1", "selected_exploratory_method": selected,
              "models": final_models, "selection_used_gold": False,
              "no_automatic_production_replacement": True,
              "inference_requires_four_prior_component_families": True, "limitations": LIMITATIONS,
              "protocol_sha256": sha_file(args.protocol)}
    save_json(args.models, models)
    result = {"status": "COMPLETE_EXPLORATORY_ORIGINAL_CPU_TRAINING", "studies": len(y),
              "report_groups": int(len(np.unique(groups))), "fold_counts": fold_counts,
              "gold_studies_excluded": gold_count, "total_excluded_studies": excluded_count,
              "metrics": metrics, "paired_group_bootstrap": bootstrap_result,
              "fold_fit_summaries": fold_fit_summaries,
              "selected_exploratory_method": selected,
              "selected_fold_ranked_delta_vs_MIL": metrics[selected]["fold_ranked_delta_vs_MIL"],
              "selected_result_interpretation": "Best of three fixed original heads for further study only; not independent CV, a competition score, or an improvement over the full Public950 ensemble.",
              "elapsed_seconds": round(time.monotonic()-start, 3),
              "clinical_rows_or_identifiers_exported": False,
              "selection_used_gold": False, "automatic_production_replacement": False,
              "limitations": LIMITATIONS, "protocol_sha256": sha_file(args.protocol),
              "models_sha256": sha_file(args.models), "source_sha256": sha_file(__file__)}
    save_json(args.results, result)
    print(json.dumps({"status": result["status"], "selected_method": selected,
                      "selected_fold_ranked_macro_auc": metrics[selected]["fold_ranked_oof"]["macro_auc"],
                      "MIL_fold_ranked_macro_auc": base["macro_auc"],
                      "selected_macro_delta": metrics[selected]["fold_ranked_delta_vs_MIL"]["macro_auc"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
