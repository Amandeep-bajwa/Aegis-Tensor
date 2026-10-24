"""Fail a pull request that changes aegis/ or src/ without changing tests/.

    python scripts/check_tests_touched.py      # in GitHub Actions, on a pull_request event

The `no-test-needed` label waives the check, but only when the lead maintainer applied it:
anyone with triage access can add a label, so the label's own events are checked through
the GitHub API (GITHUB_TOKEN with pull-requests: read and issues: read).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

CODE_DIRS = ("aegis/", "src/")
TEST_DIR = "tests/"
WAIVER_LABEL = "no-test-needed"
WAIVER_APPROVERS = {"KrishG7"}


def needs_tests(changed: list[str]) -> bool:
    """True when code changed and no test did."""
    touches_code = any(path.startswith(CODE_DIRS) for path in changed)
    touches_tests = any(path.startswith(TEST_DIR) for path in changed)
    return touches_code and not touches_tests


def waiver_applied_by(events: list[dict]) -> str | None:
    """Login of whoever applied the waiver label last, or None if it is not currently applied."""
    holder = None
    for event in events:
        if event.get("label", {}).get("name") != WAIVER_LABEL:
            continue
        if event.get("event") == "labeled":
            holder = (event.get("actor") or {}).get("login")
        elif event.get("event") == "unlabeled":
            holder = None
    return holder


def _get_all(path: str) -> list[dict]:
    """Every item of a paginated GitHub API list."""
    items: list[dict] = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}?per_page=100&page={page}",
            headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https API URL
            batch = json.load(response)
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def main() -> int:
    with open(os.environ["GITHUB_EVENT_PATH"]) as handle:
        pull = json.load(handle)["pull_request"]
    files = _get_all(f"pulls/{pull['number']}/files")
    # A renamed file counts under both its old and new path.
    changed = [f["filename"] for f in files] + [f["previous_filename"] for f in files if f.get("previous_filename")]
    if not needs_tests(changed):
        print("ok: tests changed, or no code under aegis/ or src/ changed")
        return 0
    labels = {label["name"] for label in pull.get("labels", [])}
    if WAIVER_LABEL in labels:
        applied_by = waiver_applied_by(_get_all(f"issues/{pull['number']}/events"))
        if applied_by in WAIVER_APPROVERS:
            print(f"ok: waived with `{WAIVER_LABEL}`, applied by {applied_by}")
            return 0
        print(f"`{WAIVER_LABEL}` was applied by {applied_by}; only {', '.join(sorted(WAIVER_APPROVERS))} can waive")
    code = [path for path in changed if path.startswith(CODE_DIRS)]
    print("This PR changes code but no tests:\n  " + "\n  ".join(code))
    print("Add or update a test under tests/ (written first, failing without the change).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
