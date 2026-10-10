# RSNA Knee: original experiments and reproducible inference

This project combines a complete public inference baseline with original, controlled experiments. The upstream public DINOsaur recipe scored **0.950**; the derivative in this repository has not been scored yet.

## Original prediction combiner

`scripts/fit_oof_combiner.py` compares a convex probability blend, a residual logistic head using the same finding from each model, and a larger cross-finding logistic head. It freezes the protocol before fitting, keeps Gold and excluded rows out, and reports paired group-bootstrap comparisons. Training inputs, fitted parameters, and clinical-data-derived experiment results stay outside this public repository.

Reusing predictions from previously selected checkpoints is exploratory evaluation. Weak report labels and reused folds cannot establish a competition-score gain or justify replacing the complete public ensemble. Prospective image experiments use a fresh split instead.

## Original MRI experiment

`src/rsna/spatial_mil.py` adds a spatial query for each finding before the existing slice-window MIL attention. Zero queries reproduce uniform spatial pooling; the queries retain a nonzero gradient so they can learn where evidence appears within a slice. The paired control uses identical initialization and the same image contract, labels, batches, loss, and training schedule.

The experiment uses the public 44-plane cache at a 140 mm field of view. Its three channels are neighbouring **sampled** planes, not necessarily adjacent native slices. A fresh, deterministic report/relation-group split reserves 20% as a locked holdout and uses four development folds. Gold58 and connected report/relation groups are excluded from optimization and checkpoint selection. Report grouping does not establish patient/scanner independence when those relations are unavailable.

Use the GPU trainer or standalone [Kaggle training notebook](notebooks/RSNA_Spatial_Ablation_Train.ipynb) to start a new experiment with the paired first fold. Before extending the same locked output directory, require both per-arm status receipts to show `status='complete'` and `completed_epochs=8`, with the pair recorded as complete. A successful notebook exit alone does not establish training completion. The final holdout can be opened once, after all four paired folds complete.

```bash
PYTHONPATH=src python scripts/train_spatial_ablation.py \
  --out /kaggle/working/original_spatial \
  --init /path/to/imagenet_initialization.pt --mode paired-first
# Extend a locked experiment created by these exact source files:
PYTHONPATH=src python scripts/train_spatial_ablation.py \
  --out /kaggle/working/original_spatial \
  --init /path/to/imagenet_initialization.pt --mode all
```

For an original V1 plan, use the [exact-source resume notebook](notebooks/RSNA_Spatial_Ablation_Resume.ipynb). It preserves the original trainer, model, and cache module bytes. The current monitoring trainer changes the source lock and cannot resume a V1 plan through the command above. Reattach the same six input versions and the complete previous `original_spatial` output folder, then set `RESUME_OUTPUT` to its verified Kaggle mount. Keep `MODE='all'` and `EVALUATE_HOLDOUT_ONCE=False` while completing development folds. Source, package, configuration, and input checks must pass without modifying locked receipts.

The notebook lists the required offline Kaggle inputs. Its initialization is a pinned, pure ImageNet backbone, rather than a competition-trained checkpoint. Volume integrity defaults to a clearly identified sampled fingerprint; `--full-volume-sha256` enables full hashing. Interrupted fits resume with optimizer, scheduler, scaler, and RNG state. Partial fits cannot pass the completion or holdout gates.

## Trained-family inference preparation

`src/rsna/raw_corpus44.py` retains the recovered 44-plane DICOM preprocessing reference: five fixed acquisition slots, 15–85% sampled-plane span, per-series intensity normalization before the 140 mm crop, and 336-pixel cache images. The wider public-ensemble reader cannot substitute for this training contract.

`src/rsna/spatial_serving.py` prepares the original full 34-window bags and validates complete paired-family provenance before development inference. It preserves the trainer's batch size, FP16 evaluation, valid-window encoding, slot/position metadata, and mean of per-fold probabilities. Empty study bags fail closed. The adapter does not publish a competition CSV or authorize model promotion.

Before a trained submission can be prepared, require all eight arm receipts to show eight completed epochs, all checkpoint hashes to match the locked family, and the single completed weak-label holdout evaluation to name the same selected checkpoints. A holdout-start marker without its completed receipt blocks further evaluation. Fresh raw/cache pixel parity and GPU serving qualification on deterministic label-free development studies remain required; this source preparation does not establish that either arm improves a competition score.

## Public inference baseline

- [Production notebook](notebooks/RSNA_Knee_Public950_Improved.ipynb): public anatomical mirror CoAtNet plus all three public ConvNeXt reader folds; exact source prediction arithmetic and fixed finding weights, including Lateral OA 0.52.
- [128-study qualification notebook](notebooks/RSNA_Knee_Public950_Qualification128.ipynb): complete label-free image inference, coverage, timing, and memory checks. **Never submit the qualification notebook.**

Required inference inputs: the competition, `goodpjw2008/rsna-knee-2-5d-convnext-reader`, and `nartaa/rsna-knee-publication-swa-weights-20261007`. Use T4 x2, Internet off, and the source pinned environment where available. Require `PUBLIC950_PRODUCTION_COMPLETE` plus a PASS production receipt before submitting the root `submission.csv`.

CPU fusion checks match the original for all twelve predictions. Missing components, invalid study identities, failed readers, nonfinite values, and mismatched submission contracts are rejected before atomic publication. These tests do not establish MRI accuracy, GPU health, or hidden-test runtime.

## GitHub execution

The public GitHub Actions workflow installs the official CPU PyTorch runtime and runs source, numerical, split, and optimizer tests. It needs no Kaggle credentials and does not download competition data. Standard public runners are CPU machines; full MRI training needs Kaggle GPUs or an existing GPU runner with the inputs mounted. No paid compute is configured.

For CPU numerical checks:

```bash
python -m pip install -e '.[test]'
python scripts/check_submission_contracts.py
python -m pytest -q
```

Spatial model tests additionally require `torch==2.11.0`, `torchvision==0.26.0`, and `timm==1.0.29`, as installed by the workflow. MRI images, reports, identifiers, labels, and training checkpoints remain external to this public repository.

Attribution and upstream model/data acknowledgements are retained in [THIRD_PARTY.md](THIRD_PARTY.md) and the notebooks; the Apache 2.0 license text is in `licenses/Apache-2.0.txt`.
