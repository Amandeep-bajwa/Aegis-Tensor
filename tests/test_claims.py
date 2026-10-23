"""The claims register is mechanical: every README claim has passing evidence.

See docs/claims.json and scripts/claims.py. Benchmark claims are re-derived from the
committed results; they fail when the rules or disarm code change until the benchmark
is re-run and `python scripts/claims.py --write` is applied.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("claims", ROOT / "scripts" / "claims.py")
claims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claims)


def test_every_claim_has_evidence_and_the_readme_matches_the_register():
    problems = claims.check(benchmarks=False)
    assert not problems, "\n".join(problems)


def _skip_if_pending(claim_id):
    claim = next(c for c in claims.load_register() if c["id"] == claim_id)
    if claim.get("status") == "pending":
        pytest.skip(f"{claim_id} is pending: README shows it as not yet claimed")


@pytest.mark.claim("C10")
def test_calibration_claim_matches_current_results_and_rules():
    _skip_if_pending("C10")
    values, stale = claims.calibration_values()
    assert not stale, "\n".join(stale)
    stated = next(c for c in claims.load_register() if c["id"] == "C10")["values"]
    assert stated == values, "run `python scripts/claims.py --write`"


@pytest.mark.claim("C11")
def test_disarm_claim_matches_current_results_and_code():
    _skip_if_pending("C11")
    pytest.importorskip("torch")
    values, stale = claims.disarm_values()
    assert not stale, "\n".join(stale)
    stated = next(c for c in claims.load_register() if c["id"] == "C11")["values"]
    assert stated == values, "run `python scripts/claims.py --write`"


def test_rules_fingerprint_ignores_comments_and_layout_but_not_code():
    import sys

    sys.path.insert(0, str(ROOT))
    from benchmarks.fingerprint import _normalise_python

    base = 'def f(x):\n    """Doc."""\n    return x + 1  # add one\n'
    assert _normalise_python(base) == _normalise_python('def f(x):\n\n    return x+1\n')
    assert _normalise_python(base) == _normalise_python(base.replace("# add one", "# increment"))
    assert _normalise_python(base) != _normalise_python(base.replace("x + 1", "x + 2"))
