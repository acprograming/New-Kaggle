"""Protect exact public fusion and prevent partial submission publication."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('submission_checks', ROOT / 'scripts/check_submission_contracts.py')
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)
UPSTREAM = (ROOT / 'tests/fixtures/upstream_fusion_356806064.py').read_text()
CANDIDATE = ROOT / 'notebooks/RSNA_Knee_Public950_Improved.ipynb'
FUSION = CHECKS.code_at(CANDIDATE, 10)
FINAL = CHECKS.code_at(CANDIDATE, 11)


@pytest.mark.parametrize('studies', [3, 128])
@pytest.mark.parametrize('tied', [False, True])
def test_exact_upstream_fusion(studies, tied):
    assert CHECKS.fusion_case(UPSTREAM, FUSION, studies, 20261009, tied)['status'] == 'PASS'


@pytest.mark.parametrize('scenario', [
    'valid', 'missing_reader', 'missing_reader_file', 'invalid_uid', 'duplicate_uid',
    'nonfinite', 'out_of_range', 'raptor_failure', 'raptor_incomplete',
    'zero_reader_slot', 'test_order_mismatch', 'sample_schema_mismatch',
])
def test_atomic_submission_gate(scenario):
    result = CHECKS.release_case(FUSION, FINAL, scenario)
    assert result['status'] == 'PASS'
    assert result['final_published'] == (scenario == 'valid')
