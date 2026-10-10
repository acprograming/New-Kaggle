"""Preparation helpers for the original locked spatial experiment.

No production notebook or CSV writer exists here. Fresh label-free development
raw/cache parity, exact GPU train/serve parity, complete-study qualification and
an atomic release gate are still required before a competition candidate can
be released. These helpers never open target arrays or evaluate the holdout.
The completed holdout *receipt* is checked only after all eight arms complete.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import raw_corpus44 as raw

UID = 'StudyInstanceUID'
ARMS = ('uniform', 'spatial')
FOLDS = (0, 1, 2, 3)
LOCKED_SOURCES = {
    'train_spatial_ablation.py': '98c8ea731b53ddb279fe0eb7e8ee2447384319ed890565178c54e6610687bac8',
    'spatial_mil.py': 'af00256574386af4578607ade078aee4ddf34fbb36985e4b311c5f8e2e9e59bb',
    'cache_data.py': '793f534444d19859e3c391e72f2d17179900f889565b04fe4e093ba15aa47a9f',
}
LOCKED_PACKAGES = {
    'python': '3.13.15', 'torch': '2.11.0+cu128', 'timm': '1.0.29',
    'numpy': '2.1.3', 'pandas': '2.3.3', 'scikit-learn': '1.6.1', 'cuda': '12.8',
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _constant(path, name):
    # Parse verified source without importing Torch or executing training code.
    bindings = {}
    def value(node):
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'list' and len(node.args) == 1 and not node.keywords):
            return list(value(node.args[0]))
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)
    for statement in ast.parse(Path(path).read_text()).body:
        if isinstance(statement, ast.Assign):
            names = [t.id for t in statement.targets if isinstance(t, ast.Name)]
            try:
                result = value(statement.value)
            except (ValueError, TypeError):
                if name in names:
                    raise
                continue
            bindings.update({n: result for n in names})
            if name in names:
                return result
    raise ValueError('Missing locked constant: ' + name)


@dataclass(frozen=True)
class PreparedFamily:
    root: Path
    source_root: Path
    plan: dict
    statuses: dict
    checkpoints: dict
    split: pd.DataFrame


def validate_complete_family(root, source_root):
    """Require original sources, all four pairs and the completed once-only receipt.

    ``source_root`` must be the retained original_spatial_code directory, whose
    trainer predates the monitoring-only source change. No checkpoint is loaded
    and no target/holdout-label archive is decoded by this metadata gate.
    This function's success is not a GPU or raw-pixel qualification certificate.
    """
    root, source_root = Path(root), Path(source_root)
    sources = {
        'train_spatial_ablation.py': source_root / 'scripts/train_spatial_ablation.py',
        'spatial_mil.py': source_root / 'src/rsna/spatial_mil.py',
        'cache_data.py': source_root / 'src/rsna/cache_data.py',
    }
    for name, path in sources.items():
        if _sha(path) != LOCKED_SOURCES[name]:
            raise ValueError('Original locked source differs: ' + name)
    plan = _json(root / 'plan.json')
    config = _constant(sources['cache_data.py'], 'DEFAULT_CONFIG')
    pixel = _constant(sources['cache_data.py'], 'CONTRACT')
    if (plan.get('schema') != 'rsna_spatial_ablation_plan_v1'
            or plan.get('config') != config or plan.get('pixel_contract') != pixel
            or plan.get('source_sha256') != LOCKED_SOURCES
            or config['n_folds'] != 4 or config['epochs'] != 8
            or config['eval_batch_size'] != 2):
        raise ValueError('Original plan/configuration identity differs')
    if (pixel['cache_shape'] != [44, 336, 336]
            or pixel['slot_widths'] != [s[2] for s in raw.SLOTS]
            or pixel['span'] != raw.CONTRACT['span']
            or pixel['crop_mm'] != raw.CONTRACT['crop_mm']
            or pixel['percentiles'] != raw.CONTRACT['percentiles']):
        raise ValueError('Raw pixel builder differs from training contract')
    if plan.get('gold_gradient_or_selection_rows') != 0 or plan.get('gold_rows') != 58:
        raise ValueError('Original Gold exclusion boundary differs')
    for name, key in [('split_manifest.csv', 'split_manifest_sha256'),
                      ('targets.npz', 'targets_sha256')]:
        if _sha(root / name) != plan[key]:
            raise ValueError('Locked archive bytes differ: ' + name)
    split = pd.read_csv(root / 'split_manifest.csv', dtype={UID: str}, usecols=[
        UID, 'corpus_row', 'group_sha256', 'role', 'fold', 'is_gold'])
    if (split[UID].isna().any() or split[UID].duplicated().any()
            or not np.array_equal(split.corpus_row, np.arange(len(split)))
            or not split.role.isin(['development', 'holdout', 'excluded']).all()
            or not split.is_gold.isin([True, False]).all()):
        raise ValueError('Invalid split identity or aligned row order')
    development = split.role == 'development'
    held = split.role == 'holdout'
    gold = split.is_gold == True
    if plan.get('counts') != {role: int((split.role == role).sum())
                              for role in ('development', 'holdout', 'excluded')}:
        raise ValueError('Locked split role counts differ')
    if (int(gold.sum()) != 58 or not held.any()
            or (gold & (development | held)).any()
            or set(split.loc[development, 'fold']) != set(FOLDS)
            or (split.loc[~development, 'fold'] != -1).any()
            or set(split.loc[development, 'group_sha256'])
            & set(split.loc[held | gold, 'group_sha256'])
            or split.loc[development].groupby('group_sha256').fold.nunique().max() != 1):
        raise ValueError('Gold/holdout/report-group fold isolation differs')
    parent = _json(root / 'experiment_status.json')
    if (parent.get('status') != 'all_pairs_complete'
            or parent.get('completed_pairs') != list(FOLDS)
            or parent.get('planned_pairs') != 4 or parent.get('config') != config
            or parent.get('partial_models_deployable') is not False
            or parent.get('gold_used_for_gradient_or_selection') is not False):
        raise ValueError('All four paired folds must complete before preparation')
    records = parent.get('checkpoints', [])
    identities = [(r.get('arm'), r.get('fold')) for r in records]
    expected = {(arm, fold) for arm in ARMS for fold in FOLDS}
    if len(records) != 8 or len(set(identities)) != 8 or set(identities) != expected:
        raise ValueError('Eight unique paired checkpoint records are required')
    records = {(r['arm'], r['fold']): r for r in records}
    statuses, checkpoints = {}, {}
    for arm, fold in sorted(expected):
        status = _json(root / f'{arm}_fold{fold}_status.json')
        tr = split.loc[development & (split.fold != fold), UID].tolist()
        va = split.loc[development & (split.fold == fold), UID].tolist()
        if not tr or not va:
            raise ValueError('Empty locked training or validation fold')
        contract = {
            'arm': arm, 'fold': fold, 'config': config, 'pixel_contract': pixel,
            'split_manifest_sha256': plan['split_manifest_sha256'],
            'targets_sha256': plan['targets_sha256'], 'source_sha256': LOCKED_SOURCES,
            'initialization_sha256': config['initialization_sha256'],
            'train_uid_sha256': hashlib.sha256('\n'.join(tr).encode()).hexdigest(),
            'validation_uid_sha256': hashlib.sha256('\n'.join(va).encode()).hexdigest(),
            'packages': LOCKED_PACKAGES, 'gold_gradient_or_selection_rows': 0,
            'holdout_gradient_or_selection_rows': 0,
            'selection': 'development weak-label macro-AUC; holdout never selects an epoch',
        }
        if (status.get('status') != 'complete' or status.get('completed_epochs') != 8
                or status.get('contract') != contract or len(status.get('history', [])) != 8):
            raise ValueError(f'Incomplete or changed arm: {arm} fold{fold}')
        history = status['history']
        scores = np.asarray([h.get('weak_macro_auc') for h in history], float)
        if ([h.get('epoch') for h in history] != list(range(1, 9))
                or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
                or status.get('best_weak_auc') != float(scores.max())):
            raise ValueError('Selected development history/score differs')
        record = records[arm, fold]
        name = f'{arm}_fold{fold}_best.pt'
        if Path(record.get('path', '')).name != name or _sha(root / name) != record.get('sha256'):
            raise ValueError('Selected checkpoint bytes/identity differ: ' + name)
        statuses[arm, fold], checkpoints[arm, fold] = status, root / name
    # The parent heldout_opened field remains False even after successful
    # evaluation. Check the actual completed receipt and once-only marker.
    marker = _json(root / 'holdout_once_started.json')
    receipt = _json(root / 'holdout_comparison.json')
    if (marker != {'config': config, 'pairs': list(FOLDS), 'selection_frozen': True}
            or receipt.get('schema') != 'rsna_spatial_ablation_holdout_v1'
            or receipt.get('evaluation_count') != 1 or receipt.get('pairs') != list(FOLDS)
            or receipt.get('weak_label_proxy_not_expert_accuracy') is not True
            or receipt.get('gold_used_for_gradient_selection_or_this_evaluation') is not False
            or receipt.get('rows') != int(held.sum())):
        raise ValueError('One completed locked weak-label holdout evaluation is required')
    for arm in ARMS:
        selected = receipt.get('results', {}).get(arm, {}).get('checkpoints', [])
        if selected != [{'file': f'{arm}_fold{fold}_best.pt',
                         'sha256': records[arm, fold]['sha256']} for fold in FOLDS]:
            raise ValueError('Holdout receipt differs from selected checkpoint family')
    return PreparedFamily(root, source_root, plan, statuses, checkpoints, split)


def pack_bag(volume, mask):
    """Match locked Bags(train=False): uint8 K34, slots5, zero/False padding.

    A missing/all-invalid study is an error. No prevalence or constant fallback
    is introduced for the new model. Channels are sampled cache planes.
    """
    volume, mask = np.asarray(volume), np.asarray(mask)
    if volume.shape != (44, 336, 336) or volume.dtype != np.uint8:
        raise ValueError('Expected uint8 corpus44 volume')
    if mask.shape != (44,) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected binary uint8 corpus44 mask')
    if not np.array_equal(mask, (volume.reshape(44, -1).sum(1) > 0).astype(np.uint8)):
        raise ValueError('Pixels and plane mask disagree')
    centers, slot_ids, relative = [], [], []
    offset = 0
    for slot, (_, _, width) in enumerate(raw.SLOTS):
        for center in range(offset + 1, offset + width - 1):
            if mask[center - 1:center + 2].all():
                centers.append(center)
                slot_ids.append(slot)
                relative.append(2 * (center - offset) / (width - 1) - 1)
        offset += width
    if not centers:
        raise ValueError('No eligible sampled-plane triplet; preparation fails closed')
    windows = np.zeros((34, 3, 336, 336), np.uint8)
    slots, keep, positions = np.full(34, 5, np.int64), np.zeros(34, bool), np.zeros(34, np.float32)
    count = len(centers)
    indices = np.asarray(centers, np.int64)[:, None] + np.asarray([-1, 0, 1])
    windows[:count] = volume[indices]
    slots[:count], keep[:count], positions[:count] = slot_ids, True, relative
    return windows, slots, keep, positions


def prepare_development_bags(family, uids, series_rows, series_root):
    """Read only requested development images; never read holdout or Gold images.

    Caller supplies CSV-order series rows keyed by UID. Decoder/header failures
    must be instrumented and rejected during the still-pending runtime
    qualification; the recovered builder preserves its reference zero-fill
    behavior for pixel parity rather than silently changing preprocessing.
    """
    family = validate_complete_family(family.root, family.source_root)
    uids = list(uids)
    allowed = set(family.split.loc[family.split.role == 'development', UID])
    if not uids or len(set(uids)) != len(uids) or not set(uids) <= allowed:
        raise ValueError('Qualification requires unique locked development UIDs only')
    bags = []
    for uid in uids:
        rows = series_rows.get(uid, [])
        if not rows or any(str(row.get(UID, '')) != uid for row in rows):
            raise ValueError('Missing or misaligned series metadata')
        bags.append(pack_bag(*raw.prepare_dicom_study(uid, rows, str(series_root))))
    return bags


def predict_development(family, arm, uids, bags, device='cuda:0'):
    """Development-only qualification predictions; never write files or scores.

    Uses the original FP16/batch2/chunk32 eval path and averages four probability
    votes. Success is engineering evidence, never a hidden-test score or an
    automatic promotion. Raw/cache and complete-study qualification is pending.
    """
    import torch

    if arm not in ARMS:
        raise ValueError('Unknown paired arm')
    # Recheck immutable files before using a previously returned metadata object.
    family = validate_complete_family(family.root, family.source_root)
    allowed = set(family.split.loc[family.split.role == 'development', UID])
    uids = list(uids)
    if (not uids or len(uids) != len(bags) or len(set(uids)) != len(uids)
            or not set(uids) <= allowed):
        raise ValueError('Prediction population is not unique development studies')
    runtime = {'python': platform.python_version(), **{
        p: importlib.metadata.version(p) for p in ('torch', 'timm', 'numpy', 'pandas', 'scikit-learn')},
        'cuda': str(torch.version.cuda)}
    device = torch.device(device)
    if runtime != LOCKED_PACKAGES or device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Use the original pinned CUDA runtime for initial qualification')
    path = family.source_root / 'src/rsna/spatial_mil.py'
    spec = importlib.util.spec_from_file_location('_locked_spatial_qualification', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    votes = []
    for fold in FOLDS:
        model = module.SpatialTargetMIL(family.plan['config'], arm == 'spatial', pretrained=False)
        contract = family.statuses[arm, fold]['contract']
        payload = module.load_checkpoint(model, family.checkpoints[arm, fold], contract=contract)
        if type(payload.get('epoch')) is not int or not 1 <= payload['epoch'] <= 8:
            raise ValueError('Checkpoint selected epoch differs from completed protocol')
        scores = [h['weak_macro_auc'] for h in family.statuses[arm, fold]['history']]
        selected_epoch = int(np.argmax(scores)) + 1
        if (payload['epoch'] != selected_epoch
                or payload.get('weak_macro_auc') != family.statuses[arm, fold]['best_weak_auc']):
            raise ValueError('Checkpoint is not the frozen earliest best development epoch')
        model = model.to(device).eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(bags), 2):
                batch = bags[start:start + 2]
                tensors = [torch.from_numpy(np.stack([bag[j] for bag in batch])).to(device)
                           for j in range(4)]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(*tensors, augment=False, chunk=32)
                predictions.append(logits.float().sigmoid().cpu().numpy())
        values = np.concatenate(predictions)
        if (values.shape != (len(uids), 12) or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()):
            raise ValueError('Nonfinite or incomplete study predictions')
        votes.append(values)
        del model, payload
        torch.cuda.empty_cache()
    return np.mean(np.stack(votes), axis=0)
