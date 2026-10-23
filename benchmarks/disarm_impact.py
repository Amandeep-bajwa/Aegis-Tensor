"""Measure what `aegis disarm` costs in model quality, per dtype and width.

Tasks (all weights loaded from safetensors at pinned revisions; no pickle is loaded):
  * ViT-base (float32): top-1 accuracy on an ImageNet-1k validation subset
  * BERT-base fine-tuned on SST-2 (float32): accuracy on the SST-2 validation set
  * GPT-2 (float32, and cast to float16): perplexity on WikiText-2 test
  * Qwen2.5-0.5B (bfloat16, its native dtype): perplexity on WikiText-2 test

Disarm is applied in memory with aegis.disarm.disarm_state at each width, including
widths above the per-dtype safe maximum (allow_unsafe), to show where quality breaks.

Usage:
    python benchmarks/disarm_impact.py            # all tasks, writes docs/disarm_impact.md
    python benchmarks/disarm_impact.py --quick    # fewer images/windows for a smoke run
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aegis.disarm import MANTISSA_BITS, SAFE_MAX_LOW_BITS, disarm_state, max_relative_change  # noqa: E402

CACHE = ROOT / "benchmarks" / ".hf-cache"
RESULTS_PATH = ROOT / "benchmarks" / "results" / "disarm_impact.json"
REPORT_PATH = ROOT / "docs" / "disarm_impact.md"

WIDTHS = {
    torch.float32: [0, 4, 8, 12, 16, 18, 20],
    torch.float16: [0, 2, 4, 6, 8],
    torch.bfloat16: [0, 1, 2, 3, 4, 5],
}
VIT = "google/vit-base-patch16-224"
BERT = "yoshitomo-matsubara/bert-base-uncased-sst2"
GPT2 = "openai-community/gpt2"
QWEN = "Qwen/Qwen2.5-0.5B"
IMAGENET = "hunarbatra/imagenet1k_val_3k"
SST2 = "stanfordnlp/sst2"
WIKITEXT = "Salesforce/wikitext"
TASK_LABELS = [
    "ViT-base · ImageNet-1k val subset",
    "BERT-base · SST-2 validation",
    "GPT-2 · WikiText-2 perplexity",
    "GPT-2 (cast to float16) · WikiText-2 perplexity",
    "Qwen2.5-0.5B · WikiText-2 perplexity",
]


def pinned(repo: str, repo_type: str = "model") -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo) if repo_type == "model" else api.dataset_info(repo)
    return info.sha


def parquet(repo: str, filename: str, sha: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset", revision=sha, cache_dir=str(CACHE)))


def sweep(model, dtype: torch.dtype, evaluate) -> list[dict]:
    """Evaluate ``model`` at every width for ``dtype``. Returns one row per width.

    Weights are held and disarmed in ``dtype``. A float32 ``model`` computes on the upcast
    weights, because half-precision matmuls on CPU are too slow to sweep; the upcast is exact,
    so every row measures the weights as stored in ``dtype``.
    """
    compute = {k: v.dtype for k, v in model.state_dict().items()}
    original = {k: (v.detach().to(dtype) if v.is_floating_point() else v.detach()).clone()
                for k, v in model.state_dict().items()}
    upcast = lambda state: {k: v.to(compute[k]) for k, v in state.items()}
    rows, baseline = [], None
    try:
        model.load_state_dict(upcast(original))
        for bits in WIDTHS[dtype]:
            if bits:
                disarmed, _ = disarm_state(original, low_bits=bits, allow_unsafe=True)
                model.load_state_dict(upcast(disarmed))
            started = time.time()
            metrics = evaluate(baseline)
            if baseline is None:
                baseline = metrics
            rows.append({
                "low_bits": bits,
                "safe": bits <= SAFE_MAX_LOW_BITS[dtype],
                "bound": 0.0 if bits == 0 else max_relative_change(dtype, bits),
                **{k: v for k, v in metrics.items() if not k.startswith("_")},
                "seconds": round(time.time() - started, 1),
            })
            print(f"    low_bits={bits:>2}: " + ", ".join(f"{k}={v}" for k, v in rows[-1].items()
                                                         if k not in ("low_bits", "safe", "bound", "seconds")), flush=True)
    finally:
        model.load_state_dict(upcast(original))
    return rows


# --- Tasks ------------------------------------------------------------------------

def vit_task(images: int) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, ViTForImageClassification

    sha, data_sha = pinned(VIT), pinned(IMAGENET, "dataset")
    table = parquet(IMAGENET, "data/validation-00000-of-00001.parquet", data_sha)
    rows = table.to_pylist()
    # One image per class first (ids 0, 3, 6, ...), then the rest, so small runs cover many classes.
    rows = sorted(rows, key=lambda r: (r["id"] % 3, r["id"]))[:images]
    processor = AutoImageProcessor.from_pretrained(VIT, revision=sha, cache_dir=str(CACHE))
    model = ViTForImageClassification.from_pretrained(VIT, revision=sha, cache_dir=str(CACHE), use_safetensors=True).eval()
    pixels = [processor(images=Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
                        return_tensors="pt")["pixel_values"] for r in rows]
    labels = torch.tensor([r["label"] for r in rows])

    def evaluate(baseline):
        preds = []
        with torch.inference_mode():
            for start in range(0, len(pixels), 32):
                preds.append(model(pixel_values=torch.cat(pixels[start:start + 32])).logits.argmax(-1))
        preds = torch.cat(preds)
        out = {"top1": round((preds == labels).float().mean().item() * 100, 2), "_preds": preds}
        out["agreement"] = 100.0 if baseline is None else round((preds == baseline["_preds"]).float().mean().item() * 100, 2)
        return out

    return {"task": "ViT-base · ImageNet-1k val subset", "model": VIT, "revision": sha, "data": IMAGENET,
            "data_revision": data_sha, "n": len(rows), "dtype": "float32", "metric": "top-1 accuracy (%)",
            "rows": sweep(model, torch.float32, evaluate)}


def bert_task() -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sha, data_sha = pinned(BERT), pinned(SST2, "dataset")
    table = parquet(SST2, "data/validation-00000-of-00001.parquet", data_sha)
    sentences, labels = table.column("sentence").to_pylist(), torch.tensor(table.column("label").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(BERT, revision=sha, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT, revision=sha, cache_dir=str(CACHE), use_safetensors=True).eval()
    batches = [tokenizer(sentences[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
               for i in range(0, len(sentences), 64)]

    def evaluate(baseline):
        with torch.inference_mode():
            preds = torch.cat([model(**b).logits.argmax(-1) for b in batches])
        out = {"accuracy": round((preds == labels).float().mean().item() * 100, 2), "_preds": preds}
        out["agreement"] = 100.0 if baseline is None else round((preds == baseline["_preds"]).float().mean().item() * 100, 2)
        return out

    return {"task": "BERT-base · SST-2 validation", "model": BERT, "revision": sha, "data": SST2,
            "data_revision": data_sha, "n": len(sentences), "dtype": "float32", "metric": "accuracy (%)",
            "rows": sweep(model, torch.float32, evaluate)}


def perplexity_task(repo: str, dtype: torch.dtype, windows: int, label: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sha, data_sha = pinned(repo), pinned(WIKITEXT, "dataset")
    text = "\n\n".join(parquet(WIKITEXT, "wikitext-2-raw-v1/test-00000-of-00001.parquet", data_sha)
                       .column("text").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha, cache_dir=str(CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=sha, cache_dir=str(CACHE), use_safetensors=True, dtype=torch.float32).eval()
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    length = 512
    chunks = [ids[i * length:(i + 1) * length].unsqueeze(0) for i in range(windows)]

    def evaluate(baseline):
        with torch.inference_mode():
            losses = [model(input_ids=c, labels=c).loss.float().item() for c in chunks]
        return {"perplexity": round(math.exp(sum(losses) / len(losses)), 3)}

    return {"task": label, "model": repo, "revision": sha, "data": f"{WIKITEXT} (wikitext-2-raw-v1 test)",
            "data_revision": data_sha, "n": windows * length, "dtype": str(dtype).replace("torch.", ""),
            "metric": "perplexity (lower is better)", "rows": sweep(model, dtype, evaluate)}


# --- Report -----------------------------------------------------------------------

def render(results: list[dict]) -> str:
    lines = [
        "# Disarm Impact",
        "",
        "Generated by `benchmarks/disarm_impact.py`. Weights are loaded from safetensors at the",
        "pinned revisions below; disarm is applied in memory at each width. Rows marked **safe** are at",
        "or below the per-dtype safe maximum that `aegis disarm` clamps to by default",
        f"(float32 {SAFE_MAX_LOW_BITS[torch.float32]}, float16 {SAFE_MAX_LOW_BITS[torch.float16]}, "
        f"bfloat16 {SAFE_MAX_LOW_BITS[torch.bfloat16]}). Wider rows were run with `allow_unsafe`.",
        "`bound` is the worst-case relative change of a single weight. float16 and bfloat16 weights are",
        "held and disarmed in that dtype, then upcast exactly to float32 for the forward pass.",
        "",
    ]
    for result in results:
        rows = result["rows"]
        metric_keys = [k for k in rows[0] if k not in ("low_bits", "safe", "bound", "seconds")]
        base = rows[0]
        lines += [
            f"## {result['task']} ({result['dtype']})",
            "",
            f"Model `{result['model']}` @ `{result['revision'][:12]}` · data `{result['data']}` @ "
            f"`{result['data_revision'][:12]}` · n = {result['n']} · metric: {result['metric']}",
            "",
            "| low_bits | default range | bound | " + " | ".join(metric_keys) + f" | Δ {metric_keys[0]} |",
            "|---|---|---|" + "---|" * len(metric_keys) + "---|",
        ]
        for row in rows:
            delta = row[metric_keys[0]] - base[metric_keys[0]]
            safe = "baseline" if row["low_bits"] == 0 else ("**safe**" if row["safe"] else "unsafe")
            bound = "—" if row["low_bits"] == 0 else (
                f"{row['bound']:.2%}" if row["bound"] >= 1e-4 else f"{row['bound']:.1e}")
            lines.append(f"| {row['low_bits']} | {safe} | {bound} | " + " | ".join(str(row[k]) for k in metric_keys)
                         + f" | {delta:+.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smaller evaluation sets for a smoke run")
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=16, help="512-token WikiText windows per perplexity run")
    parser.add_argument("--only", nargs="*", choices=["vit", "bert", "gpt2", "gpt2-f16", "qwen"])
    args = parser.parse_args()
    images, windows = (64, 2) if args.quick else (args.images, args.windows)
    tasks = {
        "vit": lambda: vit_task(images),
        "bert": bert_task,
        "gpt2": lambda: perplexity_task(GPT2, torch.float32, windows, TASK_LABELS[2]),
        "gpt2-f16": lambda: perplexity_task(GPT2, torch.float16, windows, TASK_LABELS[3]),
        "qwen": lambda: perplexity_task(QWEN, torch.bfloat16, windows, TASK_LABELS[4]),
    }
    previous = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() and not args.quick else []
    results = []
    for name, task in tasks.items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}", flush=True)
        results.append(task())
        if not args.quick:
            # Keep tasks from earlier runs (for example a run restricted with --only), in task order.
            done = {r["task"] for r in results}
            merged = results + [r for r in previous if r["task"] not in done]
            order = {label: i for i, label in enumerate(TASK_LABELS)}
            merged.sort(key=lambda r: order.get(r["task"], len(order)))
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(merged, indent=1) + "\n")
            REPORT_PATH.write_text(render(merged))
    print(render(results) if args.quick else f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
