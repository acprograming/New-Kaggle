"""Synthetic engineering checks; no MRI data or serving qualification claim."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rsna import raw_corpus44 as raw
from rsna import spatial_serving as serving

ROOT = Path(__file__).resolve().parents[1]


def test_sampled_planes_have_fixed_span_and_are_not_native_triplets():
    expected = [15, 21, 28, 34, 40, 46, 53, 59, 65, 71, 78, 84]
    assert raw.span_indices(100, 12).tolist() == expected
    assert np.all(np.diff(expected) > 1)
    assert raw.span_indices(1, 8).tolist() == [0] * 8
    assert raw.span_indices(2, 6).tolist() == [0] * 6
    with pytest.raises(ValueError):
        raw.span_indices(0, 12)


def test_series_selection_retains_csv_order_and_never_reuses_a_series():
    def row(uid, fluid, plane='Sagittal'):
        return {'SeriesInstanceUID': uid, 'Fluid_Sensitive': fluid, 'Anatomical_Plane': plane}
    rows = [row('z-fluid', 1), row('a-fluid', True), row('nonfluid', 0)]
    assert raw.pick_series(rows, 'Sagittal', 1, set())['SeriesInstanceUID'] == 'z-fluid'
    assert raw.pick_series(rows, 'Sagittal', 1, {'z-fluid'})['SeriesInstanceUID'] == 'a-fluid'
    assert raw.pick_series(rows, 'Sagittal', 0, set())['SeriesInstanceUID'] == 'nonfluid'
    assert raw.pick_series(rows, 'Sagittal', 0, {'nonfluid'})['SeriesInstanceUID'] == 'z-fluid'
    assert raw.pick_series(rows, 'Axial', -1, set()) is None


def test_failed_sample_keeps_its_slot_and_percentiles_precede_crop():
    contract = {**raw.CONTRACT, 'cache_shape': [6, 2, 2],
                'slots': [['Sagittal', 1, 3], ['Coronal', 0, 3]]}
    rows = [{'SeriesInstanceUID': 's', 'Fluid_Sensitive': 1, 'Anatomical_Plane': 'Sagittal'}]
    normalized_seen = []
    # Three native files select indices0,0,1. Both repeated failed samples must
    # stay zero; they must not compress the later valid sampled plane leftwards.
    def order(_):
        return [('bad', .5), ('good', .6), ('unused', .7)], .6
    def read(path):
        if path == 'bad':
            raise ValueError('Synthetic decode failure')
        assert path == 'good'
        return np.asarray([[10., 10.], [10., 110.]], np.float32)
    def resize(image, spacing, _contract):
        normalized_seen.append(image.copy())
        assert spacing == .6
        # Stand-in for a physical crop selecting the bright region. The full
        # decoded image must already have been normalized before this callback.
        return np.full((2, 2), image[1, 1])
    volume, mask = raw.prepare_dicom_study('synthetic', rows, '/unused', contract,
                                          order_fn=order, read_fn=read, resize_fn=resize)
    assert mask.tolist() == [0, 0, 1, 0, 0, 0]
    assert np.all(volume[2] == 255)
    assert np.count_nonzero(volume[:2]) == np.count_nonzero(volume[3:]) == 0
    assert len(normalized_seen) == 1
    assert np.all(normalized_seen[0][0] == 0)
    assert normalized_seen[0][1, 1] == 1


def locked_bags_class():
    # Execute only the two verified public packing definitions, using identity
    # tensor conversion. This compares APIs without importing Torch or labels.
    source = ROOT / 'src/rsna/cache_data.py'
    definitions = [n for n in ast.parse(source.read_text()).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in ('eligible_centers', 'Bags')]
    namespace = {'np': np, 'SLOT_WIDTHS': (12, 10, 8, 6, 8), 'Dataset': object,
                 'torch': SimpleNamespace(from_numpy=lambda x: x)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['Bags']


@pytest.mark.parametrize('valid_planes', [range(44), range(1, 11), [0, 1, 2, 22, 23, 24]])
def test_packing_matches_locked_evaluation_bags_and_slot_positions(valid_planes):
    mask = np.zeros(44, np.uint8)
    mask[list(valid_planes)] = 1
    volume = np.broadcast_to(np.arange(1, 45, dtype=np.uint8)[:, None, None],
                             (44, 336, 336)).copy()
    volume[mask == 0] = 0
    class Corpus:
        masks = mask[None]
        def __getitem__(self, _):
            return volume
    reference = locked_bags_class()(Corpus(), [0], np.zeros((1, 12)), np.ones((1, 12)),
                                     {}, train=False, fold=0)[0][:4]
    candidate = serving.pack_bag(volume, mask)
    for expected, actual in zip(reference, candidate):
        assert np.array_equal(expected, actual)
        assert expected.dtype == actual.dtype
    assert candidate[0].shape == (34, 3, 336, 336)
    valid = candidate[2]
    assert np.all(candidate[1][~valid] == 5)
    assert np.all(candidate[3][~valid] == 0)
    assert np.count_nonzero(candidate[0][~valid]) == 0


def test_empty_and_nonbinary_studies_fail_closed():
    volume = np.zeros((44, 336, 336), np.uint8)
    with pytest.raises(ValueError, match='fails closed'):
        serving.pack_bag(volume, np.zeros(44, np.uint8))
    with pytest.raises(ValueError, match='binary'):
        serving.pack_bag(volume, np.full(44, 2, np.uint8))
    with pytest.raises(ValueError, match='disagree'):
        serving.pack_bag(volume, np.ones(44, np.uint8))


@pytest.fixture
def synthetic_family(tmp_path):
    """Fabricated metadata/checkpoint bytes, not a loadable or qualified model."""
    code, family = tmp_path / 'code', tmp_path / 'family'
    family.mkdir()
    notebook = json.loads((ROOT / 'notebooks/RSNA_Spatial_Ablation_Resume.ipynb').read_text())
    embedded = None
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'EMBEDDED' for t in node.targets):
                    embedded = ast.literal_eval(node.value)
    assert embedded
    for name, content in embedded.items():
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    cache = code / 'src/rsna/cache_data.py'
    config, pixel = serving._constant(cache, 'DEFAULT_CONFIG'), serving._constant(cache, 'CONTRACT')
    rows = []
    for index in range(68):
        role = 'excluded' if index < 58 else ('development' if index < 66 else 'holdout')
        rows.append({serving.UID: f'synthetic-{index}', 'corpus_row': index,
                     'group_sha256': f'synthetic-group-{index}', 'role': role,
                     'fold': (index - 58) % 4 if role == 'development' else -1,
                     'is_gold': index < 58})
    split = pd.DataFrame(rows)
    split.to_csv(family / 'split_manifest.csv', index=False)
    (family / 'targets.npz').write_bytes(b'opaque synthetic archive: never decoded')
    plan = {'schema': 'rsna_spatial_ablation_plan_v1', 'config': config, 'pixel_contract': pixel,
            'source_sha256': serving.LOCKED_SOURCES,
            'split_manifest_sha256': serving._sha(family / 'split_manifest.csv'),
            'targets_sha256': serving._sha(family / 'targets.npz'),
            'gold_rows': 58, 'gold_gradient_or_selection_rows': 0,
            'counts': {'development': 8, 'holdout': 2, 'excluded': 58}}
    def write(name, value):
        (family / name).write_text(json.dumps(value))
    write('plan.json', plan)
    checkpoints, holdout_results = [], {}
    for arm in serving.ARMS:
        chosen = []
        for fold in serving.FOLDS:
            dev = split.role == 'development'
            tr = split.loc[dev & (split.fold != fold), serving.UID].tolist()
            va = split.loc[dev & (split.fold == fold), serving.UID].tolist()
            contract = {k: plan[k] for k in ('config', 'pixel_contract', 'source_sha256',
                                             'split_manifest_sha256', 'targets_sha256')}
            contract.update({'arm': arm, 'fold': fold,
                'initialization_sha256': config['initialization_sha256'],
                'train_uid_sha256': serving.hashlib.sha256('\n'.join(tr).encode()).hexdigest(),
                'validation_uid_sha256': serving.hashlib.sha256('\n'.join(va).encode()).hexdigest(),
                'packages': serving.LOCKED_PACKAGES, 'gold_gradient_or_selection_rows': 0,
                'holdout_gradient_or_selection_rows': 0,
                'selection': 'development weak-label macro-AUC; holdout never selects an epoch'})
            history = [{'epoch': i + 1, 'weak_macro_auc': s} for i, s in
                       enumerate([.6, .7, .8, .81, .8, .78, .77, .76])]
            write(f'{arm}_fold{fold}_status.json', {'status': 'complete', 'completed_epochs': 8,
                 'contract': contract, 'history': history, 'best_weak_auc': .81})
            name = f'{arm}_fold{fold}_best.pt'
            (family / name).write_bytes(f'synthetic opaque checkpoint {arm} {fold}'.encode())
            digest = serving._sha(family / name)
            checkpoints.append({'arm': arm, 'fold': fold, 'path': '/old/mount/' + name, 'sha256': digest})
            chosen.append({'file': name, 'sha256': digest})
        holdout_results[arm] = {'checkpoints': chosen}
    write('experiment_status.json', {'status': 'all_pairs_complete', 'completed_pairs': [0, 1, 2, 3],
        'planned_pairs': 4, 'config': config, 'partial_models_deployable': False,
        'gold_used_for_gradient_or_selection': False, 'heldout_opened': False, 'checkpoints': checkpoints})
    write('holdout_once_started.json', {'config': config, 'pairs': [0, 1, 2, 3], 'selection_frozen': True})
    write('holdout_comparison.json', {'schema': 'rsna_spatial_ablation_holdout_v1', 'evaluation_count': 1,
        'pairs': [0, 1, 2, 3], 'weak_label_proxy_not_expert_accuracy': True,
        'gold_used_for_gradient_selection_or_this_evaluation': False, 'rows': 2, 'results': holdout_results})
    return family, code


def test_complete_metadata_remaps_checkpoint_mount_without_opening_targets(synthetic_family):
    root, code = synthetic_family
    family = serving.validate_complete_family(root, code)
    assert len(family.checkpoints) == 8
    assert family.checkpoints['spatial', 3] == root / 'spatial_fold3_best.pt'
    assert family.plan['pixel_contract']['slot_widths'] == [12, 10, 8, 6, 8]
    assert all(s['history'][3]['weak_macro_auc'] == s['best_weak_auc'] for s in family.statuses.values())


@pytest.mark.parametrize('problem', ['partial_arm', 'missing_holdout', 'source_changed',
                                    'checkpoint_changed', 'duplicate_vote', 'history_changed', 'target_changed'])
def test_incomplete_or_changed_family_is_rejected(synthetic_family, problem):
    root, code = synthetic_family
    if problem == 'missing_holdout':
        (root / 'holdout_comparison.json').unlink()
    elif problem == 'source_changed':
        (code / 'scripts/train_spatial_ablation.py').write_text('# changed trainer')
    elif problem == 'checkpoint_changed':
        (root / 'spatial_fold3_best.pt').write_bytes(b'wrong checkpoint')
    elif problem == 'target_changed':
        (root / 'targets.npz').write_bytes(b'different archive')
    else:
        name = 'experiment_status.json' if problem == 'duplicate_vote' else 'spatial_fold3_status.json'
        value = serving._json(root / name)
        if problem == 'partial_arm':
            value['completed_epochs'] = 7
        elif problem == 'history_changed':
            value['history'][3]['epoch'] = 8
        else:
            value['checkpoints'][-1] = value['checkpoints'][0]
        (root / name).write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        serving.validate_complete_family(root, code)


def test_mutated_allowlist_cannot_expose_holdout_or_gold_images(synthetic_family, monkeypatch):
    root, code = synthetic_family
    family = serving.validate_complete_family(root, code)
    family.split.loc[family.split[serving.UID] == 'synthetic-66', 'role'] = 'development'
    def unexpected_read(*args, **kwargs):
        pytest.fail('No excluded image may be decoded')
    monkeypatch.setattr(raw, 'prepare_dicom_study', unexpected_read)
    for uid in ['synthetic-66', 'synthetic-0', 'missing']:
        with pytest.raises(ValueError, match='development'):
            serving.prepare_development_bags(family, [uid], {}, '/unused')
