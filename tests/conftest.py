"""Claim markers (see docs/claims.json and scripts/claims.py).

With AEGIS_REQUIRE_CLAIMS=1 (set in CI), a test that proves a claim must run: a skip
counts as a failure, unless the claim is limited to another platform or still pending. Otherwise a
missing optional dependency could silently leave a README claim unproved.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Scans in tests are deterministic: optional engines run only where a test asks for them.
os.environ.setdefault("AEGIS_ENGINES", "none")

REGISTER = Path(__file__).resolve().parent.parent / "docs" / "claims.json"
_CLAIMS = {c["id"]: c for c in json.loads(REGISTER.read_text())["claims"]}


def pytest_configure(config):
    config.addinivalue_line("markers", "claim(*ids): this test is evidence for the listed claims in docs/claims.json")
    config.addinivalue_line("markers", "network: tests that touch the network (deselected by default)")


def _must_run(item) -> bool:
    marker = item.get_closest_marker("claim")
    if marker is None or os.environ.get("AEGIS_REQUIRE_CLAIMS") != "1":
        return False
    claims = [_CLAIMS.get(claim_id, {}) for claim_id in marker.args]
    # A pending claim is not shown as a claim, so its test may skip.
    return any(
        c.get("status") != "pending" and (c.get("platform") is None or sys.platform.startswith(c["platform"]))
        for c in claims
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not hasattr(report, "wasxfail") and _must_run(item):
        report.outcome = "failed"
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.longrepr = f"claim test skipped while AEGIS_REQUIRE_CLAIMS=1: {reason}"
