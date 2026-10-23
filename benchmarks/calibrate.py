"""Real-world calibration: scan popular Hugging Face safetensors models with default
thresholds and report per-rule false-positive rates by dtype and model family.

These repos are assumed clean (popular, first-party or widely used uploads), so any
finding counts as a false positive. That assumption is stated in the report.

Usage:
    python benchmarks/calibrate.py                  # scan, cache downloads, write report
    python benchmarks/calibrate.py --evict          # delete each repo after scanning (low disk)
    python benchmarks/calibrate.py --report-only    # rebuild docs/calibration.md from results

Revisions are pinned by commit hash in benchmarks/calibration_lock.json on first
resolve; later runs scan exactly those revisions unless --refresh-lock is given.
Results are written after every model, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "benchmarks" / "calibration_lock.json"
RESULTS_PATH = ROOT / "benchmarks" / "results" / "calibration.json"
REPORT_PATH = ROOT / "docs" / "calibration.md"
# Separate from other benchmarks' cache, so --evict can never remove a file they are reading.
DEFAULT_CACHE = ROOT / "benchmarks" / ".hf-cache-calibration"
# Free space kept in reserve beyond a repo's own size before downloading it.
DISK_HEADROOM_BYTES = 2_000_000_000
PROPOSALS_START = "<!-- proposals:start -->"
PROPOSALS_END = "<!-- proposals:end -->"

# (repo, family). Family "quantized" covers 8-bit and 4-bit variants.
MODELS: list[tuple[str, str]] = [
    # Encoders (text)
    ("google-bert/bert-base-uncased", "encoder"),
    ("distilbert/distilbert-base-uncased", "encoder"),
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "encoder"),
    ("FacebookAI/roberta-base", "encoder"),
    ("FacebookAI/xlm-roberta-base", "encoder"),
    ("albert/albert-base-v2", "encoder"),
    ("sentence-transformers/all-MiniLM-L6-v2", "encoder"),
    ("sentence-transformers/all-mpnet-base-v2", "encoder"),
    ("BAAI/bge-small-en-v1.5", "encoder"),
    ("intfloat/e5-small-v2", "encoder"),
    ("google/electra-small-discriminator", "encoder"),
    ("answerdotai/ModernBERT-base", "encoder"),
    ("nomic-ai/nomic-embed-text-v1.5", "encoder"),
    ("microsoft/deberta-v3-small", "encoder"),
    # Vision
    ("google/vit-base-patch16-224", "vision"),
    ("microsoft/resnet-50", "vision"),
    ("facebook/convnext-tiny-224", "vision"),
    ("facebook/dinov2-small", "vision"),
    ("openai/clip-vit-base-patch32", "vision"),
    ("microsoft/swin-tiny-patch4-window7-224", "vision"),
    ("facebook/deit-small-patch16-224", "vision"),
    ("timm/resnet18.a1_in1k", "vision"),
    ("timm/mobilenetv3_large_100.ra_in1k", "vision"),
    ("timm/efficientnet_b0.ra_in1k", "vision"),
    ("facebook/detr-resnet-50", "vision"),
    ("nvidia/segformer-b0-finetuned-ade-512-512", "vision"),
    ("google/siglip-base-patch16-224", "vision"),
    # Decoder LLMs
    ("openai-community/gpt2", "decoder"),
    ("openai-community/gpt2-medium", "decoder"),
    ("distilbert/distilgpt2", "decoder"),
    ("EleutherAI/pythia-160m", "decoder"),
    ("EleutherAI/pythia-410m", "decoder"),
    ("facebook/opt-125m", "decoder"),
    ("Qwen/Qwen2.5-0.5B", "decoder"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "decoder"),
    ("Qwen/Qwen3-0.6B", "decoder"),
    ("HuggingFaceTB/SmolLM2-135M", "decoder"),
    ("HuggingFaceTB/SmolLM2-360M", "decoder"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "decoder"),
    ("bigscience/bloom-560m", "decoder"),
    ("state-spaces/mamba-130m-hf", "decoder"),
    # Audio
    ("openai/whisper-tiny", "audio"),
    ("openai/whisper-base", "audio"),
    ("openai/whisper-small", "audio"),
    ("facebook/wav2vec2-base-960h", "audio"),
    ("facebook/hubert-base-ls960", "audio"),
    ("microsoft/wavlm-base-plus", "audio"),
    ("facebook/encodec_24khz", "audio"),
    ("MIT/ast-finetuned-audioset-10-10-0.4593", "audio"),
    ("laion/clap-htsat-unfused", "audio"),
    # Encoder-decoder and multimodal
    ("google-t5/t5-small", "seq2seq"),
    ("google/flan-t5-small", "seq2seq"),
    ("facebook/bart-base", "seq2seq"),
    ("Salesforce/blip-image-captioning-base", "seq2seq"),
    # Quantized (8-bit and 4-bit)
    ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "quantized"),
    ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", "quantized"),
    ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "quantized"),
    ("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "quantized"),
    ("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8", "quantized"),
    ("unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", "quantized"),
    ("unsloth/tinyllama-bnb-4bit", "quantized"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "quantized"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ", "quantized"),
    ("RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic", "quantized"),
]


# --- Hub access ------------------------------------------------------------------

def resolve_lock(api, refresh: bool) -> dict:
    """Commit hash per repo, resolved once and then reused."""
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() and not refresh else {}
    changed = False
    for repo, _ in MODELS:
        if repo in lock:
            continue
        try:
            lock[repo] = {"sha": api.model_info(repo).sha, "error": None}
        except Exception as error:  # gated, missing or renamed repos are recorded, not fatal
            lock[repo] = {"sha": None, "error": f"{type(error).__name__}: {str(error)[:160]}"}
        changed = True
    if changed:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def repo_cache_dir(cache: Path, repo: str) -> Path:
    return cache / f"models--{repo.replace('/', '--')}"


def _linked_blobs(directory: Path) -> set[str]:
    return {os.path.realpath(p) for p in directory.rglob("*") if p.is_symlink()}


def evict_repo(cache: Path, repo: str) -> None:
    """Remove a repo from the cache, including its file contents.

    huggingface_hub 2.x keeps contents in a shared blob store at the cache root and
    only symlinks to it from the repo folder, so deleting the folder alone frees
    nothing. Blobs still linked from another repo are kept.
    """
    repo_dir = repo_cache_dir(cache, repo)
    if not repo_dir.exists():
        return
    targets = _linked_blobs(repo_dir)
    shutil.rmtree(repo_dir, ignore_errors=True)
    still_linked = set()
    for other in cache.glob("*--*"):
        still_linked |= _linked_blobs(other)
    for blob in targets - still_linked:
        Path(blob).unlink(missing_ok=True)
    for partial in cache.rglob("*.incomplete"):
        partial.unlink(missing_ok=True)


# --- Scanning ----------------------------------------------------------------------

def scan_file(path: Path) -> dict:
    from aegis import scan_safetensors
    from aegis.formatscan import scan_safetensors_header

    tensors = []
    for r in scan_safetensors(str(path)):
        tensors.append({
            "name": r.name,
            "dtype": r.dtype,
            "elements": r.num_elements,
            "rules": list(r.rules),
            "entropy": round(r.entropy, 4),
            "exponent_lane": None if r.exponent_lane_entropy is None else round(r.exponent_lane_entropy, 4),
            "low_plane": None if r.low_plane_entropy is None else round(r.low_plane_entropy, 4),
            "out_of_range": None if r.out_of_range_fraction is None else round(r.out_of_range_fraction, 6),
        })
    header = scan_safetensors_header(path)
    return {
        "tensors": tensors,
        "header_findings": [{"channel": f.channel, "where": f.where, "message": f.message} for f in header.findings],
    }


def scan_repo(api, repo: str, family: str, sha: str, cache: Path, max_gb: float, evict: bool) -> dict:
    from huggingface_hub import hf_hub_download

    record = {"repo": repo, "family": family, "sha": sha, "files": [], "skipped": None}
    info = api.model_info(repo, revision=sha, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    if not files:
        record["skipped"] = "no .safetensors files at this revision"
        return record
    total = sum(s.size or 0 for s in files)
    record["bytes"] = total
    if total > max_gb * 1e9:
        record["skipped"] = f"{total / 1e9:.1f} GB exceeds --max-gb {max_gb}"
        return record
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if free < total + DISK_HEADROOM_BYTES:
        raise DiskFull(f"{repo} needs {total / 1e9:.1f} GB plus headroom; {free / 1e9:.1f} GB free")
    try:
        for sibling in files:
            started = time.time()
            local = hf_hub_download(repo, sibling.rfilename, revision=sha, cache_dir=str(cache))
            scanned = scan_file(Path(local))
            scanned.update({"file": sibling.rfilename, "bytes": sibling.size, "seconds": round(time.time() - started, 1)})
            record["files"].append(scanned)
    finally:
        if evict:
            evict_repo(cache, repo)
    return record


class DiskFull(RuntimeError):
    """Raised before a download that would not fit; the run stops instead of filling the disk."""


# --- Aggregation and report ----------------------------------------------------------

RULES = ("pooled_entropy", "exponent_lane", "out_of_range", "low_plane", "unreadable")
SEVERITY = {"unreadable": "high", "exponent_lane": "high", "out_of_range": "high", "low_plane": "high", "pooled_entropy": "medium"}


def _pct(numerator: int, denominator: int) -> str:
    return "—" if denominator == 0 else f"{100 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def _quantiles(values: list[float]) -> str:
    if not values:
        return "—"
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return f"min {values[0]:.3f} · p1 {pick(0.01):.3f} · p50 {pick(0.5):.3f} · p99 {pick(0.99):.3f} · max {values[-1]:.3f}"


def render_report(results: list[dict], lock: dict) -> str:
    scanned = [r for r in results if r.get("files")]
    skipped = [r for r in results if not r.get("files")]
    unresolved = {repo: v["error"] for repo, v in lock.items() if v.get("sha") is None}

    by_dtype = defaultdict(lambda: {"tensors": 0, **{rule: 0 for rule in RULES}})
    by_family = defaultdict(lambda: {"tensors": 0, **{rule: 0 for rule in RULES}})
    metrics = defaultdict(lambda: defaultdict(list))
    examples = defaultdict(list)
    model_rows = []
    for record in scanned:
        model_rules = defaultdict(int)
        header_hits = 0
        tensors = 0
        dtypes = set()
        for file in record["files"]:
            header_hits += len(file["header_findings"])
            for t in file["tensors"]:
                tensors += 1
                dtypes.add(t["dtype"])
                by_dtype[t["dtype"]]["tensors"] += 1
                by_family[record["family"]]["tensors"] += 1
                for key in ("entropy", "exponent_lane", "low_plane", "out_of_range"):
                    if t[key] is not None:
                        metrics[t["dtype"]][key].append(t[key])
                for rule in t["rules"]:
                    if rule in RULES:
                        by_dtype[t["dtype"]][rule] += 1
                        by_family[record["family"]][rule] += 1
                        model_rules[rule] += 1
                        if len(examples[(rule, t["dtype"])]) < 5:
                            examples[(rule, t["dtype"])].append(f"`{record['repo']}` · `{t['name']}`")
        worst = max((SEVERITY[r] for r in model_rules), key=["medium", "high"].index, default=None)
        if header_hits and worst != "high":
            worst = "high"
        model_rows.append((record["repo"], record["family"], record["sha"][:10], ", ".join(sorted(dtypes)), tensors,
                           ", ".join(f"{k}×{v}" for k, v in sorted(model_rules.items())) or "—", header_hits, worst))

    total_tensors = sum(v["tensors"] for v in by_dtype.values())
    flagged_models = sum(1 for row in model_rows if row[-1] is not None)
    lines = [
        "# Calibration on Real Models",
        "",
        "Generated by `benchmarks/calibrate.py`. Every repo below is assumed clean, so every finding",
        "is counted as a false positive. Revisions are pinned by commit hash in",
        "`benchmarks/calibration_lock.json`. Default thresholds were used throughout.",
        "",
        "## Summary",
        "",
        f"- Repos scanned: **{len(scanned)}**; skipped: {len(skipped)}; unresolved: {len(unresolved)}",
        f"- Tensors scanned: **{total_tensors}**",
        f"- Models with any finding (would exit 1 at `--fail-on medium`): **{_pct(flagged_models, len(model_rows))}**",
        f"- Models with a high-severity finding (would exit 1 at `--fail-on high`): "
        f"**{_pct(sum(1 for r in model_rows if r[-1] == 'high'), len(model_rows))}**",
        "",
        "## Per-rule false-positive rate by dtype (tensor level)",
        "",
        "| dtype | tensors | " + " | ".join(RULES) + " |",
        "|---|---|" + "---|" * len(RULES),
    ]
    for dtype, row in sorted(by_dtype.items(), key=lambda kv: -kv[1]["tensors"]):
        lines.append(f"| {dtype} | {row['tensors']} | " + " | ".join(_pct(row[r], row["tensors"]) for r in RULES) + " |")
    lines += [
        "",
        "## Per-rule false-positive rate by model family (tensor level)",
        "",
        "| family | tensors | " + " | ".join(RULES) + " |",
        "|---|---|" + "---|" * len(RULES),
    ]
    for family, row in sorted(by_family.items()):
        lines.append(f"| {family} | {row['tensors']} | " + " | ".join(_pct(row[r], row["tensors"]) for r in RULES) + " |")
    lines += ["", "## Metric distributions by dtype", "",
              "Clean-model distributions behind each rule. Defaults: pooled entropy > 7.92 (tensors ≥ 4 KiB),",
              "exponent lane > 6.5, out-of-range fraction > 0.001, low plane < 7.5.", ""]
    for dtype in sorted(metrics):
        lines.append(f"**{dtype}**")
        lines.append("")
        for key in ("entropy", "exponent_lane", "out_of_range", "low_plane"):
            if metrics[dtype][key]:
                lines.append(f"- {key}: {_quantiles(metrics[dtype][key])} (n={len(metrics[dtype][key])})")
        lines.append("")
    if examples:
        lines += ["## Example false positives", ""]
        for (rule, dtype), names in sorted(examples.items()):
            lines.append(f"- **{rule}** on {dtype}: " + "; ".join(names))
        lines.append("")
    lines += [
        "## Per-model results",
        "",
        "| repo | family | revision | dtypes | tensors | rule hits | header findings | worst |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(model_rows, key=lambda r: (r[1], r[0])):
        lines.append("| " + " | ".join(str(x if x is not None else "clean") for x in row) + " |")
    if skipped or unresolved:
        lines += ["", "## Skipped", ""]
        for record in skipped:
            lines.append(f"- `{record['repo']}`: {record['skipped']}")
        for repo, error in sorted(unresolved.items()):
            lines.append(f"- `{repo}`: could not resolve ({error})")
    lines += ["", "## Proposals", "", PROPOSALS_START, "_Not written yet._", PROPOSALS_END, ""]
    return "\n".join(lines)


def write_report(results: list[dict], lock: dict) -> None:
    text = render_report(results, lock)
    if REPORT_PATH.exists():  # keep hand-written proposals across regenerations
        old = REPORT_PATH.read_text()
        if PROPOSALS_START in old and PROPOSALS_END in old:
            kept = old[old.index(PROPOSALS_START):old.index(PROPOSALS_END) + len(PROPOSALS_END)]
            text = text[:text.index(PROPOSALS_START)] + kept + text[text.index(PROPOSALS_END) + len(PROPOSALS_END):]
    REPORT_PATH.write_text(text)


# --- Entry point ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--evict", action="store_true", help="delete each repo's files after scanning it")
    parser.add_argument("--max-gb", type=float, default=3.0, help="skip repos whose safetensors exceed this size")
    parser.add_argument("--refresh-lock", action="store_true", help="re-resolve revisions instead of using the lock")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing results")
    parser.add_argument("--only", nargs="*", help="scan only these repos")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    lock = resolve_lock(api, args.refresh_lock) if not args.report_only else json.loads(LOCK_PATH.read_text())
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    done = {r["repo"] for r in results}

    if not args.report_only:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        for repo, family in MODELS:
            if repo in done or (args.only and repo not in args.only) or lock[repo]["sha"] is None:
                continue
            print(f"[{len(done) + 1}/{len(MODELS)}] {repo}", flush=True)
            try:
                record = scan_repo(api, repo, family, lock[repo]["sha"], args.cache_dir, args.max_gb, args.evict)
            except DiskFull as error:
                print(f"stopping: {error}. Free space or pass --evict, then re-run to resume.", flush=True)
                break
            except Exception as error:
                record = {"repo": repo, "family": family, "sha": lock[repo]["sha"], "files": [],
                          "skipped": f"error: {type(error).__name__}: {str(error)[:160]}"}
            results.append(record)
            done.add(repo)
            RESULTS_PATH.write_text(json.dumps(results, indent=1) + "\n")
            write_report(results, lock)
    write_report(results, lock)
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
