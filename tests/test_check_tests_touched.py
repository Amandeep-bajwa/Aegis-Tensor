"""scripts/check_tests_touched.py: a PR that changes code must change tests, unless waived by the lead."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_tests_touched", ROOT / "scripts" / "check_tests_touched.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _label(event, login, name="no-test-needed"):
    return {"event": event, "actor": {"login": login}, "label": {"name": name}}


def test_code_change_without_tests_needs_tests():
    assert check.needs_tests(["aegis/picklescan.py", "README.md"])
    assert check.needs_tests(["src/lib.rs"])


def test_code_change_with_tests_passes():
    assert not check.needs_tests(["aegis/picklescan.py", "tests/test_picklescan.py"])


def test_docs_or_ci_only_change_passes():
    assert not check.needs_tests(["docs/THREAT_MATRIX.md", ".github/workflows/ci.yml", "scripts/claims.py"])


def test_waiver_counts_only_while_applied_and_reports_who_applied_it():
    assert check.waiver_applied_by([]) is None
    assert check.waiver_applied_by([_label("labeled", "KrishG7")]) == "KrishG7"
    assert check.waiver_applied_by([_label("labeled", "someone-else")]) == "someone-else"
    assert check.waiver_applied_by([_label("labeled", "KrishG7"), _label("unlabeled", "KrishG7")]) is None


def test_waiver_from_lead_is_not_kept_after_someone_else_reapplies_it():
    events = [_label("labeled", "KrishG7"), _label("unlabeled", "x"), _label("labeled", "someone-else")]
    assert check.waiver_applied_by(events) == "someone-else"


def test_other_labels_are_ignored():
    assert check.waiver_applied_by([_label("labeled", "KrishG7", name="bug")]) is None
