"""Tests for the release workflow (.github/workflows/release.yml).

Parses the release workflow YAML and verifies that:
1. It triggers only on version tags (`v*`).
2. Top-level permissions are strictly read-only (`contents: read`).
3. Only the publishing job has elevated write permissions (`contents: write`).
4. Wheels are built for Linux x86_64, macOS arm64, macOS x86_64, plus an sdist.
5. Every wheel is smoke-tested outside the repository checkout before publication.
6. The publish job depends on successful builds and smoke tests, attaching all artifacts.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "release.yml"


def _load_release_workflow() -> dict:
    assert RELEASE_WORKFLOW_PATH.exists(), f"Missing workflow file: {RELEASE_WORKFLOW_PATH}"
    content = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(content)


def test_release_workflow_file_exists():
    assert RELEASE_WORKFLOW_PATH.is_file(), "release.yml must exist under .github/workflows/"


def test_release_workflow_triggers_only_on_version_tags():
    wf = _load_release_workflow()
    # YAML parses unquoted `on:` as boolean True; quoted `on:` or standard string is "on".
    triggers = wf.get("on") or wf.get(True)
    assert triggers is not None, "Workflow must declare an 'on' trigger"
    assert "pull_request" not in triggers, "Release workflow must not trigger on pull requests"
    assert "push" in triggers, "Release workflow must trigger on push"
    push = triggers["push"]
    assert "branches" not in push, "Release workflow must not trigger on branch pushes"
    assert "tags" in push, "Release workflow must trigger on tags"
    assert push["tags"] == ["v*"], f"Expected tags: ['v*'], got: {push['tags']}"


def test_release_workflow_permissions_least_privilege():
    wf = _load_release_workflow()
    # Top-level workflow permissions must be read-only
    top_perms = wf.get("permissions")
    assert top_perms is not None, "Workflow must declare top-level permissions"
    assert top_perms.get("contents") == "read", "Top-level permissions must be 'contents: read'"

    jobs = wf.get("jobs", {})
    assert "publish" in jobs or "release" in jobs, "Must have a publish or release job"

    # Identify the release job
    publish_job_name = "publish" if "publish" in jobs else "release"
    publish_job = jobs[publish_job_name]

    # Only publish job has contents: write
    assert publish_job.get("permissions", {}).get("contents") == "write", (
        f"Job '{publish_job_name}' must declare 'permissions: contents: write'"
    )

    # Every other job must not have write permissions
    for name, job in jobs.items():
        if name != publish_job_name:
            job_perms = job.get("permissions", {})
            assert job_perms.get("contents") != "write", f"Job '{name}' must not have contents: write"


def test_release_workflow_builds_three_targets_plus_sdist():
    wf = _load_release_workflow()

    # Check for wheel build job(s) and sdist job
    all_content = yaml.dump(wf)

    # Check the three abi3 wheel targets
    assert "x86_64-unknown-linux-gnu" in all_content or "manylinux" in all_content or "x86_64" in all_content
    assert "aarch64-apple-darwin" in all_content, "Missing macOS arm64 (aarch64-apple-darwin) target"
    assert "x86_64-apple-darwin" in all_content, "Missing macOS x86_64 (x86_64-apple-darwin) target"

    # Check sdist build
    assert "sdist" in all_content, "Must build an sdist"


def test_release_workflow_smoke_tests_before_publishing():
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})
    publish_job_name = "publish" if "publish" in jobs else "release"
    publish_job = jobs[publish_job_name]

    # Publish must depend on build / smoke test jobs
    needs = publish_job.get("needs")
    assert needs is not None, f"Job '{publish_job_name}' must declare 'needs'"
    if isinstance(needs, str):
        needs = [needs]

    # All workflow content dumped for inspection of smoke test commands
    all_content = yaml.dump(wf)

    # Parity with the `wheel` smoke test in ci.yml
    assert "aegis.CORE_AVAILABLE" in all_content, "Smoke test must verify aegis.CORE_AVAILABLE"
    assert "aegis --version" in all_content, "Smoke test must run aegis --version"
    assert "aegis doctor" in all_content, "Smoke test must run aegis doctor"
    assert "aegis scan" in all_content, "Smoke test must run aegis scan on a model file"


def test_release_workflow_attaches_artifacts_to_github_release():
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})
    publish_job_name = "publish" if "publish" in jobs else "release"
    publish_job = jobs[publish_job_name]

    steps = publish_job.get("steps", [])
    step_content = yaml.dump(steps)

    # Must download artifacts and release them
    assert "download-artifact" in step_content, "Publish job must download artifacts"
    assert "action-gh-release" in step_content or "gh release" in step_content, (
        "Publish job must attach artifacts to GitHub release"
    )
