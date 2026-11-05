"""Claims register: render the README's claims section and check it against the evidence.

    python scripts/claims.py            # report problems; exit 1 if any
    python scripts/claims.py --write    # refresh benchmark values in docs/claims.json and the README section

Sources of truth:
  * docs/claims.json: claim IDs and wording. Benchmark claims have a `values` dict.
  * tests marked @pytest.mark.claim("<id>", ...): the evidence for each claim.
  * benchmarks/results/*.json: the evidence for benchmark claims, with the rules
    fingerprint each result was produced under (benchmarks/fingerprint.py).

tests/test_claims.py runs check() in CI, so a claim without passing evidence fails the build.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "claims.json"
README = ROOT / "README.md"
TESTS = ROOT / "tests"
START, END = "<!-- claims:start -->", "<!-- claims:end -->"
REPO_URL = "https://github.com/KrishG7/Aegis-Tensor/blob/main"


def load_register() -> list[dict]:
    return json.loads(REGISTER.read_text(encoding="utf-8"))["claims"]


def collect_markers() -> dict[str, list[tuple[str, str, int]]]:
    """claim id -> [(test file, test name, line)] from @pytest.mark.claim decorators."""
    found: dict[str, list[tuple[str, str, int]]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "claim"
                ):
                    for arg in decorator.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
                            found.setdefault(arg.value, []).append((rel, node.name, node.lineno))
    return found


# --- Benchmark values --------------------------------------------------------------


def _import_benchmark(name: str):
    sys.path.insert(0, str(ROOT))
    import importlib

    return importlib.import_module(f"benchmarks.{name}")


def calibration_values() -> tuple[dict, list[str]]:
    calibrate = _import_benchmark("calibrate")
    fingerprint = _import_benchmark("fingerprint")
    results = json.loads(calibrate.RESULTS_PATH.read_text(encoding="utf-8"))
    numbers = calibrate.summary(results)
    current = fingerprint.scan_fingerprint()
    problems = []
    for group in ("safetensors", "mlx", "gguf", "templates"):
        stale = [f for f in numbers[group]["fingerprints"] if f != current]
        if stale:
            problems.append(
                f"calibration {group} results were produced under rules {stale}, current rules are "
                f"{current}: re-run benchmarks/calibrate.py"
            )
    values = {
        "safetensors_flagged": numbers["safetensors"]["flagged"],
        "safetensors_models": numbers["safetensors"]["models"],
        "mlx_flagged": numbers["mlx"]["flagged"],
        "mlx_models": numbers["mlx"]["models"],
        "gguf_flagged": numbers["gguf"]["flagged"],
        "gguf_models": numbers["gguf"]["models"],
        "template_flagged": numbers["templates"]["flagged_files"],
        "template_files": numbers["templates"]["files"],
    }
    return values, problems


def disarm_values() -> tuple[dict, list[str]]:
    impact = _import_benchmark("disarm_impact")
    fingerprint = _import_benchmark("fingerprint")
    from aegis.disarm import DEFAULT_LOW_BITS, default_width

    results = {r["task"]: r for r in json.loads(impact.RESULTS_PATH.read_text(encoding="utf-8"))}
    current = fingerprint.disarm_fingerprint()
    keys = dict(zip(["vit", "bert", "gpt2", "gpt2_f16", "qwen"], impact.TASK_LABELS, strict=True))
    values, problems = {}, []
    for key, label in keys.items():
        task = results.get(label)
        if task is None:
            problems.append(f"disarm benchmark has no result for {label!r}")
            continue
        if task.get("fingerprint") != current:
            problems.append(
                f"disarm result {label!r} was produced under disarm code {task.get('fingerprint')}, "
                f"current is {current}: re-run benchmarks/disarm_impact.py"
            )
        width = default_width(task["dtype"], DEFAULT_LOW_BITS)
        rows = {row["low_bits"]: row for row in task["rows"]}
        metric = next(k for k in task["rows"][0] if k not in ("low_bits", "safe", "bound", "seconds"))
        if width not in rows:
            problems.append(f"disarm result {label!r} has no row at the default width {width}")
            continue
        values[key] = f"{rows[width][metric] - rows[0][metric]:+.2f}"
    return values, problems


BENCHMARK_VALUES = {"C10": calibration_values, "C11": disarm_values}


# --- Rendering and checks ----------------------------------------------------------


def render_section(claims: list[dict], markers: dict) -> str:
    lines = [START, "", "| ID | Claim | Evidence |", "|---|---|---|"]
    for kind_label, kinds in (("claim", (None, "benchmark")), ("limit", ("limit",))):
        for claim in claims:
            if claim.get("kind") not in kinds:
                continue
            if claim.get("status") == "pending":
                text = (
                    f"_Pending: not claimed until its benchmark has run under the current code "
                    f"(see [{Path(claim['benchmark']).name}]({claim['benchmark']}))._"
                )
            else:
                text = claim["text"].format(**claim.get("values", {}))
            if claim.get("platform"):
                text += f" _(platform: {claim['platform']})_"
            evidence = []
            if claim.get("benchmark"):
                evidence.append(f"[{Path(claim['benchmark']).name}]({claim['benchmark']})")
            for file, name, _line in markers.get(claim["id"], []):
                evidence.append(f"[`{name}`]({file})")
            prefix = "**Known limit:** " if kind_label == "limit" else ""
            lines.append(f"| {claim['id']} | {prefix}{text} | {'<br>'.join(evidence)} |")
    lines += ["", END]
    return "\n".join(lines)


def check(write: bool = False, benchmarks: bool = True) -> list[str]:
    """Problems with the register, its evidence and the README section. ``benchmarks=False``
    skips re-deriving benchmark values (checked separately by their own claim tests).

    A benchmark claim with ``"status": "pending"`` is rendered as pending and claims nothing;
    ``--write`` clears the status once the results are current."""
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    claims = register["claims"]
    markers = collect_markers()
    problems = []
    ids = [c["id"] for c in claims]
    if len(ids) != len(set(ids)):
        problems.append("duplicate claim IDs in docs/claims.json")
    for claim in claims:
        if not markers.get(claim["id"]):
            problems.append(f"{claim['id']} has no test marked @pytest.mark.claim({claim['id']!r})")
    for claim_id in sorted(set(markers) - set(ids)):
        problems.append(f"tests mark claim {claim_id}, which is not in docs/claims.json")
    for claim in claims:
        if claim.get("kind") == "benchmark" and (benchmarks or write):
            values, stale = BENCHMARK_VALUES[claim["id"]]()
            if write:
                claim["values"] = values
                if stale:
                    claim["status"] = "pending"
                else:
                    claim.pop("status", None)
            if claim.get("status") == "pending":
                continue  # shown as pending in the README; nothing is claimed yet
            problems += stale
            if not write and claim.get("values") != values:
                problems.append(
                    f"{claim['id']} states {claim.get('values')}, but the results give {values}: "
                    "run `python scripts/claims.py --write`"
                )
    if write:
        REGISTER.write_text(json.dumps(register, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    section = render_section(claims, markers)
    readme = README.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        problems.append("README.md has no claims section markers")
    else:
        current = readme[readme.index(START) : readme.index(END) + len(END)]
        if current != section:
            if write:
                README.write_text(readme.replace(current, section), encoding="utf-8")
            else:
                problems.append("README claims section is out of date: run `python scripts/claims.py --write`")
    return problems


if __name__ == "__main__":
    found = check(write="--write" in sys.argv)
    for problem in found:
        print(f"- {problem}")
    sys.exit(1 if found else 0)
