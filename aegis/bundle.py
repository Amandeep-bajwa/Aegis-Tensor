"""`aegis report-bundle`: package findings for coordinated disclosure (SECURITY.md).

The bundle is a folder for the user to review and submit themselves. Aegis never sends
anything. It holds:

  REPORT.md      a draft report, including where it should go (platform or owner)
  manifest.json  file hashes and sizes, source repository and revision, verdicts
  findings.json  the findings document (docs/schema/findings-v1.schema.json)

The model file itself is never copied into the bundle: the platform already has it, and
a bundle that carries a payload would spread it. Secrets found in metadata stay masked.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.verdict import OWNER, PLATFORM

_HF_CACHE = re.compile(r"models--(?P<org>[^/]+?)--(?P<name>[^/]+)/snapshots/(?P<sha>[0-9a-f]{40})/(?P<path>.+)$")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_of(path: Path, repo_url: str | None, revision: str | None) -> dict[str, str | None]:
    """Repository URL, revision and path in the repo; read from a Hugging Face cache path when not given."""
    match = _HF_CACHE.search(Path(path).resolve().as_posix())
    if match:
        repo_url = repo_url or f"https://huggingface.co/{match['org']}/{match['name']}"
        revision = revision or match["sha"]
    return {"repo_url": repo_url, "revision": revision, "path_in_repo": match["path"] if match else Path(path).name}


def _route(verdicts: list[dict[str, Any]]) -> str:
    routes = {v["disclosure"] for v in verdicts}
    return PLATFORM if PLATFORM in routes else OWNER if OWNER in routes else "none"


def write_bundle(
    out_dir: Path,
    outcomes: list[dict[str, Any]],
    document: dict[str, Any],
    repo_url: str | None,
    revision: str | None,
    tool_version: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for outcome in outcomes:
        path = Path(outcome["path"])
        files.append(
            {
                "name": path.name,
                "size": path.stat().st_size if path.exists() else None,
                "sha256": sha256_of(path) if path.exists() else None,
                "source": source_of(path, repo_url, revision),
                "format": outcome.get("format"),
                "verdict": outcome["verdict"],
                "findings": [
                    {"rule": f.rule, "severity": f.severity, "message": f.message, "location": f.location}
                    for f in outcome["findings"]
                ],
            }
        )
    route = _route([f["verdict"] for f in files])
    manifest = {
        "tool": {"name": "aegis-tensor", "version": tool_version},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sent": False,
        "disclosure_route": route,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "findings.json").write_text(json.dumps(document, indent=2) + "\n")
    (out_dir / "REPORT.md").write_text(render_report(manifest))
    return out_dir


_WHERE = {
    PLATFORM: [
        "**Report to the platform first. Do not contact the uploader**: the owner may be the attacker, and a warning "
        "lets them re-upload elsewhere.",
        "- Hugging Face: use the **Report** option in the repository's menu, and email `security@huggingface.co` "
        "with this report and `manifest.json`.",
        "- Other hubs or mirrors: the platform's abuse or security contact.",
        "- Do not name the repository publicly until the platform has acted.",
    ],
    OWNER: [
        "**This looks accidental. Contact the owner privately** and suggest a fix (for example a safetensors "
        "version, or revoking a leaked credential). Hub discussions and pull requests are public: do not post "
        "a leaked secret there.",
    ],
    "none": ["Nothing in these results needs disclosure."],
}
_FIX = {
    "AEGIS-H004": "Revoke the credential with its issuer and remove it from the metadata.",
    "AEGIS-P002": "Publish the weights as safetensors so loading them cannot run code.",
    "AEGIS-P008": "Store the array with a numeric dtype, or publish safetensors.",
    "AEGIS-K002": "Ship the custom layer's source with the model, or avoid custom objects.",
    "AEGIS-X003": "Document the custom operator domain and where its implementation comes from.",
}


def render_report(manifest: dict[str, Any]) -> str:
    files = manifest["files"]
    sources = sorted({f["source"]["repo_url"] for f in files if f["source"]["repo_url"]})
    title = ", ".join(sources) or ", ".join(f["name"] for f in files)
    lines = [
        f"# Security report: {title}",
        "",
        f"Prepared with Aegis-Tensor {manifest['tool']['version']} on {manifest['generated_at']}.",
        "",
        "> **Draft. Nothing has been sent.** Review every section before you submit it. Static analysis can be "
        "wrong, and the decision to report is yours.",
        "",
        "## Where to send this",
        "",
        *_WHERE[manifest["disclosure_route"]],
        "",
        "## Files",
        "",
        "| File | Repository | Revision | Size (bytes) | SHA-256 | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for f in files:
        source = f["source"]
        lines.append(
            f"| `{source['path_in_repo']}` | {source['repo_url'] or 'unknown'} | "
            f"`{source['revision'] or 'unknown'}` | {f['size']} | `{f['sha256']}` | {f['verdict']['label']} |"
        )
    lines += ["", "## Findings", ""]
    for f in files:
        lines.append(f"### `{f['name']}`: {f['verdict']['label']}")
        lines.append("")
        lines.append(f"{f['verdict']['reason']}.")
        lines.append("")
        if not f["findings"]:
            lines.append("No findings.")
        for finding in f["findings"]:
            where = ", ".join(f"{k} `{v}`" for k, v in finding["location"].items() if k != "file")
            lines.append(
                f"- **{finding['rule']}** ({finding['severity']}){' at ' + where if where else ''}: "
                f"{finding['message']}"
            )
        fixes = sorted({_FIX[x["rule"]] for x in f["findings"] if x["rule"] in _FIX})
        if fixes:
            lines += ["", "Suggested fixes for the owner:", *[f"- {fix}" for fix in fixes]]
        lines.append("")
    lines += [
        "## How to reproduce",
        "",
        "1. Download the file at the revision above and check its SHA-256.",
        f"2. `pip install aegis-tensor=={manifest['tool']['version']}`",
        "3. `aegis scan <file> --format json`. Aegis reads the file without loading or executing it.",
        "",
        "`findings.json` is the full machine-readable result (schema: "
        "https://github.com/KrishG7/Aegis-Tensor/blob/main/docs/schema/findings-v1.schema.json).",
        "",
    ]
    return "\n".join(lines)


def default_bundle_dir(paths: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"aegis-report-{Path(paths[0]).stem}-{stamp}")
