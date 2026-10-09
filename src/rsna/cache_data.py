"""Public corpus44 inputs, confidence-weighted targets, and a prospective split.

Actual inputs:
  dreaddevelopment/knee-raptor-corpus: all_ids.npy, all_masks.npy, all_vols.npy
  dreaddevelopment/knee-raptor-corpus-ext: extra_ids.npy, extra_masks.npy, extra_vols.npy
The uint8 volumes have 44 sampled planes at 336 pixels. Neighbouring cached
planes need not be neighbouring native DICOM slices. Triplets never cross slots.
Only generated local experiment directories contain IDs or label arrays.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
SLOT_WIDTHS = (12, 10, 8, 6, 8)
CONTRACT = {'name': 'dread_corpus44_slot_safe_spatial_v1', 'cache_shape': [44, 336, 336],
            'cache_dtype': 'uint8', 'slot_widths': list(SLOT_WIDTHS),
            'span': [.15, .85], 'crop_mm': 140., 'percentiles': [2., 98.],
            'cache_resize': 'cv2.INTER_AREA', 'uint8_conversion': 'multiply_255_then_truncate',
            'triplets': 'three positive-mask sampled planes inside one slot', 'max_windows': 34,
            'positions': '2*(center-slot_start)/(slot_width-1)-1', 'padding_slot': 5,
            'native_slice_adjacency_established': False}
DEFAULT_CONFIG = {'architecture': 'convnext_tiny.fb_in22k_ft_in1k', 'resolution': 320,
                  'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                  'gradient_checkpointing': True, 'head_dropout': .2, 'spatial_key_dim': 64,
                  'seed': 20261009, 'n_folds': 4, 'holdout_fraction': .20, 'epochs': 8,
                  'train_windows': 24, 'batch_size': 2, 'eval_batch_size': 2, 'workers': 2,
                  'backbone_lr': 3e-5, 'head_lr': 1e-3, 'weight_decay': .02,
                  'positive_weight_cap': 5., 'label_smoothing': .02, 'gradient_clip': 3.,
                  'max_hours': 8.5, 'reserve_minutes': 20,
                  'initialization_sha256': '6270cae1e860a267e3b9a6f73932d627edaf3529a32bf9b80e79f04c936aa7b9'}
DEFAULT_TEACHERS = [('rsna-knee-hybrid-report-labels', 'report_labels_v4hybrid.csv'),
                    ('rsna-knee-llm-report-labels', 'llm_labels_v4_blend.csv')]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()



def volume_fingerprint(path, full=False):
    """Record exact size and either a full or explicitly sampled content digest.

    Full hashing is opt-in: these public arrays total roughly22GB. Sampled
    hashing checks the header and16evenly-spaced64KiBblocks; it is not a proof
    that every pixel byte is unchanged. Immutable Kaggle dataset inputs should
    be reused; callers may enable full hashing for stronger local provenance.
    """
    path=Path(path)
    if full:
        return {'bytes':path.stat().st_size,'method':'full_sha256','sha256':sha256(path)}
    size=path.stat().st_size
    h=hashlib.sha256()
    offsets=np.linspace(0,max(0,size-65536),16).round().astype(np.int64)
    with path.open('rb') as f:
        for offset in offsets:
            f.seek(int(offset)); h.update(str(int(offset)).encode()); h.update(f.read(65536))
    return {'bytes':size,'method':'sampled_16x64KiB_sha256_not_full_pixel_verification','sha256':h.hexdigest()}

def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def find_dataset_file(slug: str, filename: str, input_root='/kaggle/input') -> str:
    root = Path(input_root)
    candidates = [root / slug] + list(root.glob('datasets/*/' + slug)) + list(root.glob('notebooks/*/' + slug))
    hits = sorted({str(p.resolve()) for d in candidates if d.is_dir() for p in d.rglob(filename) if p.is_file()})
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected one {slug}/{filename}; found {hits}')
    return hits[0]


def discover_inputs(input_root='/kaggle/input'):
    paths = {}
    for suffix, slug, prefix in [('a', 'knee-raptor-corpus', 'all'), ('b', 'knee-raptor-corpus-ext', 'extra')]:
        for kind in ('ids', 'masks', 'vols'):
            paths[kind + '_' + suffix] = find_dataset_file(slug, prefix + '_' + kind + '.npy', input_root)
    root = Path(input_root)
    train = [p for p in (root / 'rsna-knee-abnormality-detection' / 'train.csv',
                        root / 'competitions' / 'rsna-knee-abnormality-detection' / 'train.csv') if p.is_file()]
    if len(train) != 1:
        raise FileNotFoundError('Attach competition train.csv exactly once')
    teachers = [find_dataset_file(s, f, input_root) for s, f in DEFAULT_TEACHERS]
    return paths, str(train[0]), teachers


class Corpus:
    def __init__(self, paths: dict[str, str], expected_counts=(3200, 1207)):
        self.paths = dict(paths)
        self.expected_counts = expected_counts
        self.arrays = [np.load(paths['vols_' + s], mmap_mode='r', allow_pickle=False) for s in ('a', 'b')]
        ids = [np.load(paths['ids_' + s], allow_pickle=False).astype(str) for s in ('a', 'b')]
        masks = [np.load(paths['masks_' + s], allow_pickle=False) for s in ('a', 'b')]
        if expected_counts is not None and tuple(len(a) for a in self.arrays) != tuple(expected_counts):
            raise ValueError('Wrong public corpus sizes; expected 3200 + 1207 studies')
        for volume, uid, mask in zip(self.arrays, ids, masks):
            if volume.shape != (len(uid), 44, 336, 336) or volume.dtype != np.uint8:
                raise ValueError('Expected uint8 study volumes N×44×336×336')
            if mask.shape != (len(uid), 44) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
                raise ValueError('Expected binary uint8 N×44 masks aligned to the same IDs')
        self.ids, self.masks = np.concatenate(ids), np.concatenate(masks)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate corpus UIDs; never independently sort arrays')
        self.boundary = len(self.arrays[0])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, row):
        return self.arrays[0][row] if row < self.boundary else self.arrays[1][row - self.boundary]
    def __getstate__(self):
        return self.paths, self.expected_counts
    def __setstate__(self, state):
        self.__init__(*state)


def eligible_centers(mask):
    mask = np.asarray(mask)
    if mask.shape != (44,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a binary 44-plane corpus mask')
    rows, offset = [], 0
    for slot, width in enumerate(SLOT_WIDTHS):
        for center in range(offset + 1, offset + width - 1):
            if mask[center - 1:center + 2].all():
                rows.append((center, slot, 2 * (center - offset) / (width - 1) - 1))
        offset += width
    return np.asarray(rows, np.float64).reshape(-1, 3)


class Bags(Dataset):
    def __init__(self, corpus, rows, targets, weights, config, *, train, fold):
        self.corpus, self.rows = corpus, np.asarray(rows, dtype=np.int64)
        self.targets, self.weights, self.config = targets, weights, config
        self.train, self.fold, self.epoch = train, fold, 0
        self.k = int(config['train_windows']) if train else 34
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = int(self.rows[index])
        table = eligible_centers(self.corpus.masks[row])
        if not len(table):
            raise ValueError(f'All-invalid cached study at row {row}')
        if self.train and len(table) > self.k:
            rng = np.random.default_rng(self.config['seed'] + self.fold * 1000003 + self.epoch * 10007 + row)
            first = [int(rng.choice(np.flatnonzero(table[:, 1] == s))) for s in np.unique(table[:, 1])]
            if len(first) > self.k:
                raise ValueError('train_windows must cover each available slot')
            remaining = np.setdiff1d(np.arange(len(table)), first)
            take = first + rng.choice(remaining, self.k - len(first), replace=False).tolist()
            table = table[np.sort(take)]
        n = min(len(table), self.k)
        windows = np.zeros((self.k, 3, 336, 336), np.uint8)
        slots, positions, mask = np.full(self.k, 5, np.int64), np.zeros(self.k, np.float32), np.zeros(self.k, bool)
        centers = table[:n, 0].astype(np.int64)
        windows[:n] = self.corpus[row][centers[:, None] + np.array([-1, 0, 1])]
        slots[:n], positions[:n], mask[:n] = table[:n, 1].astype(np.int64), table[:n, 2], True
        return (torch.from_numpy(windows), torch.from_numpy(slots), torch.from_numpy(mask),
                torch.from_numpy(positions), torch.from_numpy(self.targets[row]),
                torch.from_numpy(self.weights[row]), row)


def group_split(ids, reports, gold, *, seed, n_folds, holdout_fraction, relations=(), excluded_ids=()):
    """Split report/known-relation components; gold-connected components are excluded.

    Optional relation pairs should include known patient/study links. Without
    these, report groups do not establish patient or scanner isolation.
    """
    if not (0 < holdout_fraction < .5) or n_folds < 2:
        raise ValueError('Invalid holdout fraction or fold count')
    ids = np.asarray(ids).astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs')
    reports = [re.sub(r'\s+', ' ', str(s).lower()).strip() for s in reports]
    parent = {u: u for u in ids}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen = {}
    for uid, report in zip(ids, reports):
        if report:
            if report in seen:
                union(uid, seen[report])
            else:
                seen[report] = uid
    for a, b in relations:
        a, b = str(a), str(b)
        if a not in parent or b not in parent:
            raise ValueError('Relation references UID outside the supplied training corpus')
        union(a, b)
    components = {}
    for uid in ids:
        components.setdefault(find(uid), []).append(uid)
    group_hash = {root: hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()
                  for root, members in components.items()}
    groups = np.asarray([group_hash[find(u)] for u in ids])
    excluded_ids = set(map(str, excluded_ids))
    if excluded_ids - set(ids):
        raise ValueError('Exclusion list references unknown corpus UID')
    forbidden = {groups[i] for i, u in enumerate(ids) if gold[i] or u in excluded_ids or not reports[i]}
    roles, folds = [], []
    for group in groups:
        digest = hashlib.sha256(f'holdout:{seed}:{group}'.encode()).hexdigest()
        if group in forbidden:
            role, fold = 'excluded', -1
        elif int(digest[:16], 16) / 2**64 < holdout_fraction:
            role, fold = 'holdout', -1
        else:
            role = 'development'
            fold = int(hashlib.sha256(f'fold:{seed}:{group}'.encode()).hexdigest()[:16], 16) % n_folds
        roles.append(role); folds.append(fold)
    return np.asarray(groups), np.asarray(roles), np.asarray(folds, np.int16)


def make_plan(config, out_dir, *, paths, train_csv, teacher_csvs, relations_csv=None, excluded_csv=None):
    """Lock the split before training. Existing plans are verified, never overwritten."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(paths)
    train = pd.read_csv(train_csv, dtype={UID: str})
    if train[UID].duplicated().any() or set(train[UID]) != set(corpus.ids):
        raise ValueError('Official train table and cache UIDs differ')
    train = train.set_index(UID).loc[corpus.ids]
    observed = train[TARGETS].notna().sum(1).to_numpy()
    if not np.isin(observed, [0, len(TARGETS)]).all():
        raise ValueError('Official expert row is partially labelled')
    gold = observed == len(TARGETS)
    if gold.sum() != 58:
        raise ValueError('Expected exactly 58 official expert-labelled studies')
    reports = train['Report'].fillna('').astype(str).tolist()
    relations = []
    if relations_csv:
        frame = pd.read_csv(relations_csv, dtype=str)
        relations = list(zip(frame[UID], frame['RelatedStudyInstanceUID']))
    excluded = []
    if excluded_csv:
        excluded = pd.read_csv(excluded_csv, usecols=[UID], dtype=str)[UID].tolist()
    groups, roles, folds = group_split(corpus.ids, reports, gold, seed=config['seed'], n_folds=config['n_folds'],
                                      holdout_fraction=config['holdout_fraction'], relations=relations, excluded_ids=excluded)
    sources, confidence, label_receipts = [], [], []
    if not teacher_csvs:
        raise ValueError('At least one weak-label source is required')
    for path in teacher_csvs:
        frame = pd.read_csv(path, dtype={UID: str})
        if frame[UID].duplicated().any():
            raise ValueError('Duplicate weak-label source UID')
        frame = frame.set_index(UID).reindex(corpus.ids)
        values = frame[TARGETS].to_numpy(np.float32)
        known = np.isfinite(values)
        if ((values[known] < 0) | (values[known] > 1)).any():
            raise ValueError('Weak probabilities must be in [0,1]')
        sources.append(values)
        cols = [t + '__conf' for t in TARGETS]
        c = frame[cols].to_numpy(np.float32) if all(k in frame for k in cols) else np.full_like(values, .75)
        c = np.nan_to_num(c, nan=.25)
        if ((c < 0) | (c > 1)).any():
            raise ValueError('Weak-label confidence must be in [0,1]')
        confidence.append(np.clip(c, .25, 1.))
        label_receipts.append({'path':str(path),'sha256':sha256(path)})
    cube = np.stack(sources)
    count = np.isfinite(cube).sum(0)
    y = np.nansum(cube, axis=0) / np.maximum(count, 1)
    disagreement = np.nansum(np.abs(cube - y[None]), axis=0) / np.maximum(count, 1)
    weights = (np.clip(1 - 2 * disagreement, .25, 1) * np.mean(confidence, axis=0)).astype(np.float32)
    weights[count == 0] = 0
    usable = np.array([len(eligible_centers(m)) > 0 for m in corpus.masks])
    roles[~usable | ~(count > 0).all(1)] = 'excluded'
    folds[roles != 'development'] = -1
    y[gold], weights[gold] = .5, 0
    if not (roles == 'holdout').any() or not (roles == 'development').any():
        raise ValueError('Hash split produced an empty holdout/development partition')
    for fold in range(config['n_folds']):
        tr = (roles == 'development') & (folds != fold)
        va = (roles == 'development') & (folds == fold)
        if not tr.any() or not va.any():
            raise ValueError('Hash split produced an empty training/validation fold')
        if set(groups[tr]) & set(groups[va]) or set(groups[tr | va]) & set(groups[gold | (roles == 'holdout')]):
            raise ValueError('Group isolation violation')
        for j, target in enumerate(TARGETS):
            hard = y[va, j][np.abs(y[va, j] - .5) > 1e-6] > .5
            if hard.sum() < 2 or (~hard).sum() < 2:
                raise ValueError(f'Fixed fold {fold} lacks both weak-label classes for {target}; do not re-roll seed after inspection')
    inputs = {'train_csv':{'path':str(train_csv),'sha256':sha256(train_csv)},'teachers':label_receipts,
              'alignment':{k:sha256(p) for k,p in paths.items() if not k.startswith('vols')},
              'volumes':{k:volume_fingerprint(p,config.get('full_volume_sha256',False)) for k,p in paths.items() if k.startswith('vols')},
              'relations':None if not relations_csv else sha256(relations_csv),
              'exclusions':None if not excluded_csv else sha256(excluded_csv)}
    source_paths = [Path(__file__), Path(__file__).with_name('spatial_mil.py')]
    trainer = Path(__file__).parents[2] / 'scripts' / 'train_spatial_ablation.py'
    if trainer.is_file():
        source_paths.append(trainer)
    source_hashes = {p.name:sha256(p) for p in source_paths}
    lock = out / 'plan.json'
    if lock.exists():
        plan = json.loads(lock.read_text())
        if plan['config'] != config or plan['inputs'] != inputs or plan['source_sha256'] != source_hashes:
            raise ValueError('Existing locked plan differs from inputs, source, or configuration')
        if sha256(out/'split_manifest.csv') != plan['split_manifest_sha256'] or sha256(out/'targets.npz') != plan['targets_sha256']:
            raise ValueError('Locked split or targets changed')
        return plan
    manifest = pd.DataFrame({UID:corpus.ids,'corpus_row':np.arange(len(corpus)),'group_sha256':groups,
                             'role':roles,'fold':folds,'is_gold':gold,'usable_pixels':usable})
    manifest.to_csv(out/'split_manifest.csv', index=False)
    np.savez_compressed(out/'targets.npz', ids=corpus.ids, targets=y.astype(np.float32), weights=weights,
                        roles=roles, folds=folds, groups=groups, gold=gold)
    plan = {'schema':'rsna_spatial_ablation_plan_v1','config':config,'pixel_contract':CONTRACT,
            'corpus_paths':paths,'inputs':inputs,'source_sha256':source_hashes,
            'split_manifest_sha256':sha256(out/'split_manifest.csv'),'targets_sha256':sha256(out/'targets.npz'),
            'counts':{k:int((roles==k).sum()) for k in ('development','holdout','excluded')},
            'gold_rows':int(gold.sum()),'gold_gradient_or_selection_rows':0,
            'validation_boundary':'Fresh report/known-relation components. No patient/scanner isolation is established without relation inputs. Holdout never selects checkpoints; weak labels are a proxy.'}
    atomic_json(plan, lock)
    return plan
