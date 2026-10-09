"""Exercise actual notebook fusion and final release cells without GPU or Kaggle writes.

Only literal /kaggle/working paths are remapped into disposable directories.
This is numerical/release-contract validation, not model accuracy or GPU qualification.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import scipy


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"


def code_at(path, index):
    notebook = json.loads(Path(path).read_text())
    cell = notebook["cells"][index]
    assert cell["cell_type"] == "code"
    return "".join(cell["source"])


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


class RedirectKaggleWorking(ast.NodeTransformer):
    def __init__(self, destination):
        self.destination = str(destination)
        self.replacements = 0

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/kaggle/working/"):
            self.replacements += 1
            return ast.copy_location(ast.Constant(self.destination + node.value[len("/kaggle/working"):]), node)
        return node


def run_cell(source, env, label):
    tree = ast.parse(source, filename=label)
    redirect = RedirectKaggleWorking(env["WORK"])
    tree = ast.fix_missing_locations(redirect.visit(tree))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(tree, label, "exec"), env)
    return {"redirected_path_literals": redirect.replacements, "stdout": stream.getvalue()}


def synthetic_tables(n, seed, tied):
    rng = np.random.default_rng(seed)
    ids = [f"synthetic-study-{i:04d}" for i in range(n)]
    aa, bb = rng.random((n, len(TARGETS))), rng.random((n, len(TARGETS)))
    if tied:
        aa, bb = np.floor(aa * 4) / 4, np.floor(bb * 4) / 4
        # Guarantee ties even in a three-study sample, including a fully tied column.
        aa[:, 0] = bb[:, 0] = 0.5
    anchor = pd.DataFrame(aa, columns=TARGETS)
    reader = pd.DataFrame(bb, columns=TARGETS)
    anchor.insert(0, UID, ids)
    reader.insert(0, UID, ids)
    return ids, anchor, reader


def environment(work, ids, anchor, reader):
    return {"ACTIVE_MODE": "apex950", "WORK": Path(work), "TARGETS": TARGETS,
            "ids": list(ids), "anchor": anchor.copy(), "reader": reader.copy(),
            "np": np, "pd": pd, "time": time, "os": os, "hashlib": hashlib,
            "T0": time.monotonic()}


def fusion_case(upstream_source, candidate_source, n, seed, tied):
    ids, anchor, reader = synthetic_tables(n, seed, tied)
    with tempfile.TemporaryDirectory(prefix="rsna-fusion-original-") as original, \
         tempfile.TemporaryDirectory(prefix="rsna-fusion-hardened-") as improved:
        left, right = environment(original, ids, anchor, reader), environment(improved, ids, anchor, reader)
        run_cell(upstream_source, left, "upstream-fusion")
        run_cell(candidate_source, right, "hardened-fusion")
        lm, rm = left["result"][TARGETS].to_numpy(), right["result"][TARGETS].to_numpy()
        assert np.array_equal(lm, rm), "Healthy candidate fusion changed"
        assert np.array_equal(left["control"][TARGETS].to_numpy(), right["control"][TARGETS].to_numpy())
        assert np.array_equal(left["EXPERIMENT_W"]["Lateral OA"], 0.52)
        assert left["EXPERIMENT_W"] == right["EXPERIMENT_W"]
        assert left["result"][UID].tolist() == right["result"][UID].tolist() == ids
        lc = pd.read_csv(Path(original) / "submission.csv")
        rc = pd.read_csv(Path(improved) / "_submission_pending.csv")
        assert np.array_equal(lc[TARGETS].to_numpy(), rc[TARGETS].to_numpy())
        assert not (Path(improved) / "submission.csv").exists(), "Fusion published final before release gate"
        return {"status": "PASS", "studies": n, "seed": seed, "tied_predictions": tied,
                "all_12_targets_exact": True, "control_exact": True,
                "CSV_roundtrip_exact": True, "max_abs_difference": float(np.max(np.abs(lm-rm))),
                "final_not_published_before_gate": True}


def release_case(candidate_source, final_source, mode, n=3):
    ids, anchor, reader = synthetic_tables(n, 314159, True)
    with tempfile.TemporaryDirectory(prefix="rsna-release-gate-") as dirname:
        work = Path(dirname)
        data = work / "data"
        data.mkdir()
        pd.DataFrame({UID: ids}).to_csv(data / "test.csv", index=False)
        sample = anchor.copy()
        sample.to_csv(data / "sample_submission.csv", index=False)
        env = environment(work, ids, anchor, reader)
        env["find_test_root"] = lambda: str(data)
        run_cell(candidate_source, env, "candidate-fusion")
        reader.to_csv(work / "_convnext_3fold.csv", index=False)
        diag = pd.DataFrame({"valid_slots": [4] * n, "attempted_slots": [6] * n,
                             "series_read_errors": [0] * n, "short_series_slots": [0] * n,
                             "bad_dicom_files": [0] * n}, index=ids)
        env["reader_diag"] = diag
        health = {"failures": [], "completed": n}
        pending = work / "_submission_pending.csv"
        if mode == "missing_reader":
            env["reader"] = None
        elif mode == "missing_reader_file":
            (work / "_convnext_3fold.csv").unlink()
        elif mode == "invalid_uid":
            frame = pd.read_csv(pending)
            frame.loc[0, UID] = "unexpected-study"
            frame.to_csv(pending, index=False)
        elif mode == "duplicate_uid":
            frame = pd.read_csv(pending)
            frame.loc[0, UID] = frame.loc[1, UID]
            frame.to_csv(pending, index=False)
        elif mode == "nonfinite":
            frame = pd.read_csv(pending)
            frame.loc[0, "ACL"] = np.nan
            frame.to_csv(pending, index=False)
        elif mode == "out_of_range":
            frame = pd.read_csv(pending)
            frame.loc[0, "ACL"] = 1.5
            frame.to_csv(pending, index=False)
        elif mode == "raptor_failure":
            health["failures"] = ["synthetic-study-0000"]
        elif mode == "raptor_incomplete":
            health["completed"] = n - 1
        elif mode == "zero_reader_slot":
            diag.loc[ids[0], "valid_slots"] = 0
        elif mode == "test_order_mismatch":
            pd.DataFrame({UID: list(reversed(ids))}).to_csv(data / "test.csv", index=False)
        elif mode == "sample_schema_mismatch":
            sample.drop(columns=["ACL"]).to_csv(data / "sample_submission.csv", index=False)
        else:
            assert mode == "valid"
        (work / "raptor_health_receipt.json").write_text(json.dumps(health))
        error = None
        try:
            run_cell(final_source, env, "final-atomic-release")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        published = (work / "submission.csv").exists()
        if mode == "valid":
            assert error is None, error
            assert published and not pending.exists(), "Valid pending candidate was not atomically promoted"
            receipt = json.loads((work / "production_receipt.json").read_text())
            final_hash = hashlib.sha256((work / "submission.csv").read_bytes()).hexdigest()
            assert receipt["submission_sha256"] == final_hash
            assert receipt["actual_leaderboard_score"] is None
            assert receipt["rows"] == n and receipt["reader_folds"] == 3
        else:
            assert error is not None, f"Fault {mode} was not rejected"
            assert not published, f"Fault {mode} left final submission.csv"
            assert not (work / "production_receipt.json").exists(), f"Fault {mode} emitted PASS receipt"
        return {"status": "PASS", "scenario": mode, "final_published": published,
                "rejection": error, "valid_atomic_rename_and_hash_verified": mode == "valid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--upstream", type=Path, default=root / "tests/fixtures/upstream_fusion_356806064.py")
    parser.add_argument("--candidate", type=Path, default=root / "notebooks/RSNA_Knee_Public950_Improved.ipynb")
    parser.add_argument("--out", type=Path, default=root / "results/submission_contracts.json")
    args = parser.parse_args()
    upstream_fusion = args.upstream.read_text() if args.upstream.suffix == ".py" else code_at(args.upstream, 9)
    candidate_fusion = code_at(args.candidate, 10)
    final_source = code_at(args.candidate, 11)
    fusion_results = [fusion_case(upstream_fusion, candidate_fusion, n, seed, tied)
                      for n in (3, 128) for seed in (42, 20261009) for tied in (False, True)]
    scenarios = ("valid", "missing_reader", "missing_reader_file", "invalid_uid", "duplicate_uid",
                 "nonfinite", "out_of_range", "raptor_failure", "raptor_incomplete", "zero_reader_slot",
                 "test_order_mismatch", "sample_schema_mismatch")
    release_results = [release_case(candidate_fusion, final_source, mode) for mode in scenarios]
    receipt = {"status": "PASS_CPU_FUSION_PARITY_AND_ATOMIC_RELEASE_GATES",
               "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
               "upstream_sha256": hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
               "source_cell_sha256": {"upstream_fusion_cell9": source_hash(upstream_fusion),
                                      "candidate_fusion_cell10": source_hash(candidate_fusion),
                                      "candidate_release_cell11": source_hash(final_source)},
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
               "fusion_cases": fusion_results, "release_cases": release_results,
               "limitation": "Synthetic CPU numerical and release-contract tests only; no MRI accuracy, checkpoint loading, GPU qualification, or hidden-runtime guarantee.",
               "kaggle_files_written": False}
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"status": receipt["status"], "fusion_cases": len(fusion_results),
                      "release_cases": len(release_results), "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
