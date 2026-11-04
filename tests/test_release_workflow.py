"""Tests for the release workflow (.github/workflows/release.yml).

Parses the release workflow YAML and verifies that:
1. It triggers only on version tags (`v*`).
2. Top-level permissions are strictly read-only (`contents: read`).
3. Only the publishing job has elevated write permissions (`contents: write`).
4. Wheels are built for the exact set: Linux x86_64, macOS arm64, macOS x86_64, plus an sdist.
5. Every wheel is smoke-tested inside the wheel job, after install, before upload.
6. The publish job depends on both the wheel job and the sdist job.

Each test must be verifiable by removing the thing it covers from the workflow and
confirming the test fails (see CONTRIBUTING.md).
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "release.yml"

EXPECTED_WHEEL_TARGETS = {
    "x86_64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
}


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
    assert "publish" in jobs, "Must have a 'publish' job"
    publish_job = jobs["publish"]

    # Only publish job has contents: write
    assert publish_job.get("permissions", {}).get("contents") == "write", (
        "Job 'publish' must declare 'permissions: contents: write'"
    )

    # Every other job must not have write permissions
    for name, job in jobs.items():
        if name != "publish":
            job_perms = job.get("permissions", {})
            assert job_perms.get("contents") != "write", f"Job '{name}' must not have contents: write"


def test_release_workflow_builds_exact_three_targets_plus_sdist():
    """Assert the exact set of wheel targets — not a loose substring match."""
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})

    # Collect targets from the matrix
    found_targets: set[str] = set()
    for _job_name, job in jobs.items():
        matrix = job.get("strategy", {}).get("matrix", {})
        includes = matrix.get("include", [])
        for entry in includes:
            if "target" in entry:
                found_targets.add(entry["target"])

    assert found_targets == EXPECTED_WHEEL_TARGETS, (
        f"Wheel targets mismatch.\n  Expected: {sorted(EXPECTED_WHEEL_TARGETS)}\n  Got:      {sorted(found_targets)}"
    )

    # sdist job must exist separately
    assert "sdist" in jobs, "Must have a separate 'sdist' job"


def test_release_workflow_smoke_tests_inside_wheel_job_before_upload():
    """Smoke-test steps must be inside the wheel job, after install, before upload-artifact."""
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})
    assert "wheels" in jobs, "Must have a 'wheels' job"
    wheel_job = jobs["wheels"]
    steps = wheel_job.get("steps", [])

    step_names = [s.get("name", "") for s in steps]
    step_uses = [s.get("uses", "") for s in steps]

    # Find the positions of install, smoke test, and upload
    install_idx = next(
        (i for i, name in enumerate(step_names) if "install" in name.lower() and "wheel" in name.lower()), None
    )
    smoke_idx = next((i for i, name in enumerate(step_names) if "smoke" in name.lower()), None)
    upload_idx = next((i for i, uses in enumerate(step_uses) if "upload-artifact" in uses), None)

    assert install_idx is not None, "Wheel job must have an 'install wheel' step"
    assert smoke_idx is not None, "Wheel job must have a 'smoke test' step"
    assert upload_idx is not None, "Wheel job must have an 'upload-artifact' step"

    assert install_idx < smoke_idx, "Smoke test must come after the install step"
    assert smoke_idx < upload_idx, "Smoke test must come before the upload step"

    # Verify smoke test content
    smoke_step = steps[smoke_idx]
    run_script = smoke_step.get("run", "")
    assert "aegis.CORE_AVAILABLE" in run_script, "Smoke test must verify aegis.CORE_AVAILABLE"
    assert "aegis --version" in run_script, "Smoke test must run aegis --version"
    assert "aegis doctor" in run_script, "Smoke test must run aegis doctor"
    assert "aegis scan" in run_script, "Smoke test must run aegis scan on a model file"

    # working-directory must be outside the checkout (runner.temp)
    working_dir = smoke_step.get("working-directory", "")
    assert "runner.temp" in working_dir or "RUNNER_TEMP" in working_dir, (
        "Smoke test must run from outside the checkout directory"
    )


def test_release_workflow_publish_needs_wheels_and_sdist():
    """publish must declare 'needs' containing both the wheel job and the sdist job."""
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})
    assert "publish" in jobs, "Must have a 'publish' job"
    publish_job = jobs["publish"]

    needs = publish_job.get("needs")
    assert needs is not None, "Job 'publish' must declare 'needs'"
    if isinstance(needs, str):
        needs = [needs]

    assert "wheels" in needs, f"'publish' must need 'wheels', got: {needs}"
    assert "sdist" in needs, f"'publish' must need 'sdist', got: {needs}"


def test_release_workflow_attaches_artifacts_to_github_release():
    wf = _load_release_workflow()
    jobs = wf.get("jobs", {})
    assert "publish" in jobs, "Must have a 'publish' job"
    publish_job = jobs["publish"]

    steps = publish_job.get("steps", [])
    step_content = yaml.dump(steps)

    # Must download artifacts and release them
    assert "download-artifact" in step_content, "Publish job must download artifacts"
    assert "action-gh-release" in step_content or "gh release" in step_content, (
        "Publish job must attach artifacts to GitHub release"
    )
