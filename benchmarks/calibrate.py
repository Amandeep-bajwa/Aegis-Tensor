"""Real-world calibration: scan popular Hugging Face models with default thresholds and
report per-rule false-positive rates by dtype, model family and format.

Three sets:
  * safetensors repos (encoders, vision, decoders, audio, quantized, and MLX models)
  * GGUF files scanned in full (quantized and F16/BF16): tensor rules on float tensors
    plus the chat-template check
  * GGUF chat templates from popular GGUF repos, read with HTTP range requests, so
    only the file header is downloaded

These repos are assumed clean (popular, first-party or widely used uploads), so any
finding counts as a false positive. That assumption is stated in the report.

Usage:
    python benchmarks/calibrate.py                  # scan, cache downloads, write report
    python benchmarks/calibrate.py --evict          # delete each repo after scanning (low disk)
    python benchmarks/calibrate.py --report-only    # rebuild docs/calibration.md from results

Revisions are pinned by commit hash in benchmarks/calibration_lock.json on first
resolve; later runs scan exactly those revisions unless --refresh-lock is given.
Results are written after every model, so an interrupted run resumes where it stopped.
Each record stores the rules fingerprint (benchmarks/fingerprint.py) it was scanned under;
records from older rules are rescanned.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aegis.thresholds import DEFAULT_THRESHOLDS  # noqa: E402
from benchmarks.fingerprint import scan_fingerprint  # noqa: E402

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
    # MLX (Apple silicon; safetensors with 4-bit or 8-bit packed uint32 weights and bf16/f16 scales)
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "mlx"),
    ("mlx-community/Qwen3-0.6B-4bit", "mlx"),
    ("mlx-community/Qwen3-0.6B-8bit", "mlx"),
    ("mlx-community/Qwen3-0.6B-bf16", "mlx"),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "mlx"),
    ("mlx-community/Llama-3.2-1B-Instruct-bf16", "mlx"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "mlx"),
    ("mlx-community/gemma-3-1b-it-4bit", "mlx"),
    ("mlx-community/gemma-3-1b-it-qat-4bit", "mlx"),
    ("mlx-community/SmolLM2-360M-Instruct", "mlx"),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "mlx"),
    ("mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", "mlx"),
    ("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", "mlx"),
    ("mlx-community/Qwen3-4B-4bit", "mlx"),
]

# GGUF files scanned in full: (repo, file). Quantized files still carry F32 tensors (norms,
# biases), which get the tensor rules; quantized blocks are skipped.
GGUF_FULL: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("unsloth/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-F16.gguf"),
    ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
    ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-BF16.gguf"),
    ("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    ("bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF", "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"),
    ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
    ("ibm-granite/granite-3.3-2b-instruct-GGUF", "granite-3.3-2b-instruct-Q4_K_M.gguf"),
    ("lmstudio-community/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
]

# GGUF chat templates only (header read by range request): (repo, file). The files in
# GGUF_FULL have their templates checked too.
GGUF_HEADER_ONLY: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
    ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    ("lmstudio-community/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    ("NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"),
    ("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"),
    ("unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"),
    ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
    ("unsloth/phi-4-GGUF", "phi-4-Q4_K_M.gguf"),
    ("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q4_K_M.gguf"),
    ("lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
    ("bartowski/granite-3.1-2b-instruct-GGUF", "granite-3.1-2b-instruct-Q4_K_M.gguf"),
    ("ggml-org/SmolLM3-3B-GGUF", "SmolLM3-Q4_K_M.gguf"),
    ("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf"),
]


# --- Hub access ------------------------------------------------------------------


def resolve_lock(api, refresh: bool) -> dict:
    """Commit hash per repo, resolved once and then reused."""
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() and not refresh else {}
    changed = False
    for repo in dict.fromkeys([r for r, _ in MODELS] + [r for r, _ in GGUF_FULL + GGUF_HEADER_ONLY]):
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

    t = DEFAULT_THRESHOLDS
    results = scan_safetensors(str(path), t["entropy"], t["benford"], t["exponent"], t["plane"])
    header = scan_safetensors_header(path)
    return {
        "tensors": [_tensor_record(r) for r in results],
        "header_findings": [
            {"channel": f.channel, "where": f.where, "message": f.message, "severity": HEADER_SEVERITY[f.channel]}
            for f in header.findings
        ],
    }


def _tensor_record(r) -> dict:
    return {
        "name": r.name,
        "dtype": r.dtype,
        "elements": r.num_elements,
        "rules": list(r.rules),
        "entropy": round(r.entropy, 4),
        "exponent_lane": None if r.exponent_lane_entropy is None else round(r.exponent_lane_entropy, 4),
        "low_plane": None if r.low_plane_entropy is None else round(r.low_plane_entropy, 4),
        "out_of_range": None if r.out_of_range_fraction is None else round(r.out_of_range_fraction, 6),
    }


def _container_findings(result) -> list[dict]:
    from aegis.findings import RULE_BY_CODE
    from aegis.findings import RULES as CATALOG

    return [
        {
            "channel": f.code,
            "where": f.location,
            "message": f.message,
            "severity": "critical" if f.severity == "CRITICAL" else CATALOG[RULE_BY_CODE[f.code]]["severity"],
        }
        for f in result.findings
    ]


def scan_gguf_file(path: Path) -> dict:
    from aegis.modelformats import scan_model_format

    result = scan_model_format(path, "gguf", DEFAULT_THRESHOLDS)
    return {
        "tensors": [_tensor_record(r) for r in result.tensor_results],
        "header_findings": _container_findings(result),
    }


def scan_gguf_header(repo: str, filename: str, sha: str) -> dict:
    """Chat-template check from the file header alone, fetched with range requests."""
    from huggingface_hub import HfFileSystem

    from aegis.modelformats import ModelFormatResult, _check_template, chat_templates, parse_gguf_header

    size = 8 << 20
    with HfFileSystem().open(f"{repo}@{sha}/{filename}", "rb", block_size=size) as handle:
        data = handle.read(size)
        while True:
            try:
                metadata, _infos, end = parse_gguf_header(data)
                break
            except ValueError as error:
                if ("truncated" not in str(error) and "past the end" not in str(error)) or size >= 256 << 20:
                    raise
                more = handle.read(size)
                size *= 2
                if not more:
                    raise
                data += more
    result = ModelFormatResult(path=filename, format="gguf")
    templates = chat_templates(metadata)
    for key, text in templates.items():
        _check_template(text, result, f"{filename}:{key}")
    return {
        "header_bytes": end,
        "templates": {k: len(v) for k, v in templates.items()},
        "tensors": [],
        "header_findings": _container_findings(result),
    }


def scan_repo(api, repo: str, family: str, sha: str, cache: Path, max_gb: float, evict: bool) -> dict:
    from huggingface_hub import hf_hub_download

    record = {"repo": repo, "family": family, "kind": "safetensors", "sha": sha, "files": [], "skipped": None}
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
            scanned.update(
                {"file": sibling.rfilename, "bytes": sibling.size, "seconds": round(time.time() - started, 1)}
            )
            record["files"].append(scanned)
    finally:
        if evict:
            evict_repo(cache, repo)
    return record


def scan_gguf_repo(api, repo: str, filename: str, sha: str, cache: Path, evict: bool, header_only: bool) -> dict:
    from huggingface_hub import hf_hub_download

    record = {
        "repo": repo,
        "family": "gguf",
        "kind": "gguf_header" if header_only else "gguf",
        "file": filename,
        "sha": sha,
        "files": [],
        "skipped": None,
    }
    info = api.model_info(repo, revision=sha, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    if filename not in sizes:
        record["skipped"] = f"{filename} not found at this revision"
        return record
    started = time.time()
    if header_only:
        scanned = scan_gguf_header(repo, filename, sha)
    else:
        free = shutil.disk_usage(cache.parent if not cache.exists() else cache).free
        if free < sizes[filename] + DISK_HEADROOM_BYTES:
            raise DiskFull(f"{repo}/{filename} needs {sizes[filename] / 1e9:.1f} GB plus headroom")
        cache.mkdir(parents=True, exist_ok=True)
        try:
            scanned = scan_gguf_file(Path(hf_hub_download(repo, filename, revision=sha, cache_dir=str(cache))))
        finally:
            if evict:
                evict_repo(cache, repo)
    scanned.update({"file": filename, "bytes": sizes[filename], "seconds": round(time.time() - started, 1)})
    record["files"].append(scanned)
    return record


class DiskFull(RuntimeError):
    """Raised before a download that would not fit; the run stops instead of filling the disk."""


# --- Aggregation and report ----------------------------------------------------------

RULES = ("pooled_entropy", "exponent_lane", "out_of_range", "low_plane", "unreadable")
SEVERITY = {
    "unreadable": "high",
    "exponent_lane": "high",
    "out_of_range": "high",
    "low_plane": "high",
    "pooled_entropy": "medium",
}
HEADER_SEVERITY = {"metadata": "high", "header_whitespace": "medium", "tensor_name": "medium", "secret": "medium"}
LEVELS = ["low", "medium", "high", "critical"]
GROUPS = {
    "safetensors": "safetensors (non-MLX)",
    "mlx": "MLX safetensors",
    "gguf": "GGUF, full scan",
}


def record_key(record: dict) -> tuple:
    return record.get("kind", "safetensors"), record["repo"], record.get("file")


def _group(record: dict) -> str:
    kind = record.get("kind", "safetensors")
    return "mlx" if kind == "safetensors" and record["family"] == "mlx" else kind


def model_outcome(record: dict) -> dict:
    """Per-model rule hits and worst severity at default thresholds."""
    rules, other, tensors, dtypes = defaultdict(int), [], 0, set()
    for file in record["files"]:
        other += file["header_findings"]
        for t in file["tensors"]:
            tensors += 1
            dtypes.add(t["dtype"])
            for rule in t["rules"]:
                if rule in RULES:
                    rules[rule] += 1
    severities = [SEVERITY[r] for r in rules] + [f.get("severity", "high") for f in other]
    worst = max(severities, key=LEVELS.index, default=None)
    return {"rules": dict(rules), "other": other, "tensors": tensors, "dtypes": sorted(dtypes), "worst": worst}


def summary(results: list[dict]) -> dict:
    """Headline numbers per group, used by the report and by tests/test_claims.py."""
    out = {}
    for group in GROUPS:
        records = [r for r in results if r.get("files") and _group(r) == group]
        outcomes = [model_outcome(r) for r in records]
        out[group] = {
            "models": len(records),
            "tensors": sum(o["tensors"] for o in outcomes),
            "flagged": sum(1 for o in outcomes if o["worst"] is not None),
            "flagged_high": sum(1 for o in outcomes if o["worst"] in ("high", "critical")),
            "fingerprints": sorted({r.get("fingerprint") for r in records}, key=str),
        }
    templates = [r for r in results if r.get("files") and r.get("kind") in ("gguf", "gguf_header")]
    template_findings = [
        f for r in templates for f in r["files"][0]["header_findings"] if f["channel"].startswith("gguf_template")
    ]
    out["templates"] = {
        "files": len(templates),
        "templates": sum(len(r["files"][0].get("templates") or {}) for r in templates),
        "flagged_files": len({f["where"].split(":")[0] for f in template_findings}),
        "critical": sum(1 for f in template_findings if f["severity"] == "critical"),
        "fingerprints": sorted({r.get("fingerprint") for r in templates}, key=str),
    }
    return out


def _pct(numerator: int, denominator: int) -> str:
    return "—" if denominator == 0 else f"{100 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def _quantiles(values: list[float]) -> str:
    if not values:
        return "—"
    values = sorted(values)

    def pick(q):
        return values[min(len(values) - 1, int(q * (len(values) - 1)))]

    return (
        f"min {values[0]:.3f} · p1 {pick(0.01):.3f} · p50 {pick(0.5):.3f} · p99 {pick(0.99):.3f} · max {values[-1]:.3f}"
    )


def render_report(results: list[dict], lock: dict) -> str:
    scanned = [r for r in results if r.get("files")]
    skipped = [r for r in results if not r.get("files")]
    unresolved = {repo: v["error"] for repo, v in lock.items() if v.get("sha") is None}
    tensor_records = [r for r in scanned if r.get("kind", "safetensors") in ("safetensors", "gguf")]

    by_dtype = defaultdict(lambda: {"tensors": 0, **{rule: 0 for rule in RULES}})
    by_family = defaultdict(lambda: {"tensors": 0, **{rule: 0 for rule in RULES}})
    metrics = defaultdict(lambda: defaultdict(list))
    examples = defaultdict(list)
    model_rows = []
    for record in tensor_records:
        family = record["family"]
        for file in record["files"]:
            for t in file["tensors"]:
                by_dtype[t["dtype"]]["tensors"] += 1
                by_family[family]["tensors"] += 1
                for key in ("entropy", "exponent_lane", "low_plane", "out_of_range"):
                    if t[key] is not None:
                        metrics[t["dtype"]][key].append(t[key])
                for rule in t["rules"]:
                    if rule in RULES:
                        by_dtype[t["dtype"]][rule] += 1
                        by_family[family][rule] += 1
                        if len(examples[(rule, t["dtype"])]) < 5:
                            examples[(rule, t["dtype"])].append(f"`{record['repo']}` · `{t['name']}`")
        o = model_outcome(record)
        name = record["repo"] + (f" · `{record['files'][0]['file']}`" if record.get("kind") == "gguf" else "")
        model_rows.append(
            (
                name,
                family,
                record["sha"][:10],
                ", ".join(o["dtypes"]),
                o["tensors"],
                ", ".join(f"{k}×{v}" for k, v in sorted(o["rules"].items())) or "—",
                ", ".join(sorted({f["channel"] for f in o["other"]})) or "—",
                o["worst"],
            )
        )

    numbers = summary(results)
    total_tensors = sum(v["tensors"] for v in by_dtype.values())
    lines = [
        "# Calibration on Real Models",
        "",
        "Generated by `benchmarks/calibrate.py`. Every file below is assumed clean, so every finding",
        "is counted as a false positive. Revisions are pinned by commit hash in",
        "`benchmarks/calibration_lock.json`. Default thresholds were used throughout",
        f"(`aegis/thresholds.py`); rules fingerprint `{scan_fingerprint()}` (`benchmarks/fingerprint.py`).",
        "",
        "## Summary",
        "",
        f"- Tensors scanned: **{total_tensors}**; skipped: {len(skipped)}; unresolved: {len(unresolved)}",
        "",
        "| set | models | tensors | any finding (exit 1 at `--fail-on medium`) | high or critical |",
        "|---|---|---|---|---|",
    ]
    for group, label in GROUPS.items():
        g = numbers[group]
        lines.append(
            f"| {label} | {g['models']} | {g['tensors']} | {_pct(g['flagged'], g['models'])} | "
            f"{_pct(g['flagged_high'], g['models'])} |"
        )
    t = numbers["templates"]
    lines += [
        "",
        "## GGUF chat templates",
        "",
        f"{t['templates']} chat templates from {t['files']} GGUF files ({len(GGUF_FULL)} scanned in full,",
        "the rest read from the file header only). Every template is from a popular, first-party or",
        "widely used upload, so any finding is a false positive.",
        "",
        f"- Files with a template finding: **{_pct(t['flagged_files'], t['files'])}**",
        f"- Critical template findings: **{t['critical']}**",
        "",
    ]
    for record in sorted((r for r in scanned if r.get("kind") in ("gguf", "gguf_header")), key=lambda r: r["repo"]):
        file = record["files"][0]
        hits = [f for f in file["header_findings"] if f["channel"].startswith("gguf_template")]
        if hits:
            for f in hits:
                lines.append(f"- `{record['repo']}` · `{f['where']}`: {f['message']}")
    lines += [
        "",
        "## Per-rule false-positive rate by dtype (tensor level)",
        "",
        "| dtype | tensors | " + " | ".join(RULES) + " |",
        "|---|---|" + "---|" * len(RULES),
    ]
    for dtype, row in sorted(by_dtype.items(), key=lambda kv: -kv[1]["tensors"]):
        lines.append(
            f"| {dtype} | {row['tensors']} | " + " | ".join(_pct(row[r], row["tensors"]) for r in RULES) + " |"
        )
    lines += [
        "",
        "## Per-rule false-positive rate by model family (tensor level)",
        "",
        "| family | tensors | " + " | ".join(RULES) + " |",
        "|---|---|" + "---|" * len(RULES),
    ]
    for family, row in sorted(by_family.items()):
        lines.append(
            f"| {family} | {row['tensors']} | " + " | ".join(_pct(row[r], row["tensors"]) for r in RULES) + " |"
        )
    lines += [
        "",
        "## Metric distributions by dtype",
        "",
        "Clean-model distributions behind each rule. Defaults: pooled entropy > 7.92 (tensors ≥ 4 KiB),",
        "exponent lane > 6.5, out-of-range fraction > 0.001, low plane < 7.5.",
        "",
    ]
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
        "| model | family | revision | dtypes | tensors | rule hits | other findings | worst |",
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
            kept = old[old.index(PROPOSALS_START) : old.index(PROPOSALS_END) + len(PROPOSALS_END)]
            text = text[: text.index(PROPOSALS_START)] + kept + text[text.index(PROPOSALS_END) + len(PROPOSALS_END) :]
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
    parser.add_argument(
        "--sets",
        nargs="*",
        choices=["safetensors", "gguf", "gguf_header"],
        default=["safetensors", "gguf", "gguf_header"],
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    lock = resolve_lock(api, args.refresh_lock) if not args.report_only else json.loads(LOCK_PATH.read_text())
    current = scan_fingerprint()
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    results = [r for r in results if r.get("fingerprint") == current or r.get("skipped")]  # stale rules: rescan
    done = {record_key(r) for r in results}

    jobs = (
        [("safetensors", repo, family) for repo, family in MODELS]
        + [("gguf", repo, file) for repo, file in GGUF_FULL]
        + [("gguf_header", repo, file) for repo, file in GGUF_HEADER_ONLY]
    )
    if not args.report_only:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        for index, (kind, repo, extra) in enumerate(jobs, 1):
            key = (kind, repo, None if kind == "safetensors" else extra)
            if (
                key in done
                or kind not in args.sets
                or (args.only and repo not in args.only)
                or lock.get(repo, {}).get("sha") is None
            ):
                continue
            sha = lock[repo]["sha"]
            print(f"[{index}/{len(jobs)}] {kind} {repo}" + (f" {extra}" if kind != "safetensors" else ""), flush=True)
            try:
                if kind == "safetensors":
                    record = scan_repo(api, repo, extra, sha, args.cache_dir, args.max_gb, args.evict)
                else:
                    record = scan_gguf_repo(api, repo, extra, sha, args.cache_dir, args.evict, kind == "gguf_header")
            except DiskFull as error:
                print(f"stopping: {error}. Free space or pass --evict, then re-run to resume.", flush=True)
                break
            except Exception as error:
                record = {
                    "repo": repo,
                    "family": extra if kind == "safetensors" else "gguf",
                    "kind": kind,
                    "sha": sha,
                    "files": [],
                    "file": None if kind == "safetensors" else extra,
                    "skipped": f"error: {type(error).__name__}: {str(error)[:160]}",
                }
            record["fingerprint"] = current
            results = [r for r in results if record_key(r) != record_key(record)] + [record]
            done.add(record_key(record))
            RESULTS_PATH.write_text(json.dumps(results, indent=1) + "\n")
            write_report(results, lock)
    write_report(results, lock)
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
