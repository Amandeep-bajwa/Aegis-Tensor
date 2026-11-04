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
    assert _normalise_python(base) == _normalise_python("def f(x):\n\n    return x+1\n")
    assert _normalise_python(base) == _normalise_python(base.replace("# add one", "# increment"))
    assert _normalise_python(base) != _normalise_python(base.replace("x + 1", "x + 2"))


# --- Negative tests for claims checker ---------------------------------------------


def _setup_tmp_claims_env(tmp_path, monkeypatch):
    """Copy docs/claims.json, README.md and a minimal test file into tmp_path."""
    import json

    reg_copy = tmp_path / "claims.json"
    readme_copy = tmp_path / "README.md"
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    reg_copy.write_text(claims.REGISTER.read_text(encoding="utf-8"), encoding="utf-8")
    readme_copy.write_text(claims.README.read_text(encoding="utf-8"), encoding="utf-8")

    register = json.loads(reg_copy.read_text(encoding="utf-8"))
    test_lines = ["import pytest\n\n"]
    for claim in register["claims"]:
        test_lines.append(f"@pytest.mark.claim({claim['id']!r})\ndef test_{claim['id']}(): pass\n\n")
    (tests_dir / "test_dummy.py").write_text("".join(test_lines), encoding="utf-8")

    monkeypatch.setattr(claims, "REGISTER", reg_copy)
    monkeypatch.setattr(claims, "README", readme_copy)
    monkeypatch.setattr(claims, "TESTS", tests_dir)
    return reg_copy, readme_copy, tests_dir


def test_check_reports_readme_out_of_date(tmp_path, monkeypatch):
    import json

    reg_copy, _, _ = _setup_tmp_claims_env(tmp_path, monkeypatch)
    data = json.loads(reg_copy.read_text(encoding="utf-8"))
    data["claims"][0]["text"] += " (tampered)"
    reg_copy.write_text(json.dumps(data, indent=2), encoding="utf-8")
    problems = claims.check(benchmarks=False)
    assert any("README claims section is out of date" in p for p in problems), problems


def test_check_reports_claim_without_a_test(tmp_path, monkeypatch):
    import json

    reg_copy, _, _ = _setup_tmp_claims_env(tmp_path, monkeypatch)
    data = json.loads(reg_copy.read_text(encoding="utf-8"))
    data["claims"].append({"id": "C999", "text": "Uncovered claim text"})
    reg_copy.write_text(json.dumps(data, indent=2), encoding="utf-8")
    problems = claims.check(benchmarks=False)
    assert any("has no test marked" in p for p in problems), problems


def test_check_reports_marker_for_unknown_claim(tmp_path, monkeypatch):
    _, _, tests_dir = _setup_tmp_claims_env(tmp_path, monkeypatch)
    (tests_dir / "test_extra.py").write_text(
        "import pytest\n@pytest.mark.claim('C99')\ndef test_c99(): pass\n",
        encoding="utf-8",
    )
    problems = claims.check(benchmarks=False)
    assert any("not in docs/claims.json" in p for p in problems), problems


def test_check_reports_duplicate_ids(tmp_path, monkeypatch):
    import json

    reg_copy, _, _ = _setup_tmp_claims_env(tmp_path, monkeypatch)
    data = json.loads(reg_copy.read_text(encoding="utf-8"))
    dup = dict(data["claims"][0])
    data["claims"].append(dup)
    reg_copy.write_text(json.dumps(data, indent=2), encoding="utf-8")
    problems = claims.check(benchmarks=False)
    assert any("duplicate claim IDs" in p for p in problems), problems


def test_sweep_accepts_torch_dtype():
    """sweep() must accept a real torch dtype, not a string.

    Regression guard: vit_task and bert_task previously passed "float32" (a
    string), which made tensor.to(dtype) raise RuntimeError. This test calls
    sweep with torch.float32 and a minimal model so the benchmark keeps working.
    """
    import sys

    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT))
    from benchmarks.disarm_impact import sweep

    model = torch.nn.Linear(4, 2)
    calls: list = []

    def evaluate(baseline):
        calls.append(1)
        return {"metric": 1.0}

    rows = sweep(model, torch.float32, evaluate)
    assert rows, "sweep returned no rows"
    assert all(isinstance(r["low_bits"], int) for r in rows), "low_bits must be int"
    assert len(calls) == len(rows), "evaluate was not called for every width"


def test_claims_script_runs_without_optional_dependencies():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['huggingface_hub'] = None\n"
        "import runpy\n"
        "runpy.run_path('scripts/claims.py', run_name='__main__')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
