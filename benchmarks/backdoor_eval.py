"""Backdoor detection benchmark on CIFAR-10: Aegis fuzzer vs ART Neural Cleanse and ART STRIP.

Models: a small CNN (global average pooling before the classifier, so it converts
exactly to Keras) trained with one recipe, either clean or poisoned by:
  * BadNets      3x3 white patch, bottom-right corner, 5% poison, label flipped to target
  * Blended      fixed noise pattern blended at alpha 0.2, 5% poison, label flipped
  * WaNet        fixed warping field (k=4, s=0.5), 10% poison + 20% noise-warped "cross" samples
  * Clean-label  Turner et al.: 50% of target-class images PGD-perturbed against a clean
                 reference model, then four 3x3 corner patches; labels are never changed

Detectors:
  * Aegis fuzzer (model level): DynamicTrojanFuzzer automated suite, default threshold.
  * ART Neural Cleanse (model level): Keras copy of each model, outlier detection on clean data.
  * ART STRIP (input level): flags triggered *inputs*; reported separately because it needs
    triggered inputs at test time and is not a model-level detector.

Model-level TPR counts only backdoored models whose attack actually implanted (ASR at least
the --min-asr value); failed implants are listed, not counted. Results are written after every
model, so a long run resumes.

Needs torch, tensorflow, adversarial-robustness-toolbox and aegis in one environment
(TensorFlow has no Python 3.14 wheels; Python 3.12 works).

Usage:
    python benchmarks/backdoor_eval.py                       # full run, writes docs/backdoor_eval.md
    python benchmarks/backdoor_eval.py --seeds 1 --clean-seeds 1 --epochs 2 --nc-steps 20   # smoke run
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "benchmarks" / ".hf-cache"
CIFAR_REPO = "uoft-cs/cifar10"
RESULTS_PATH = ROOT / "benchmarks" / "results" / "backdoor_eval.json"
REPORT_PATH = ROOT / "docs" / "backdoor_eval.md"
MODEL_DIR = ROOT / "benchmarks" / "results" / "backdoor_models"
TARGET = 0
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
ATTACKS = ("badnets", "blended", "wanet", "clean_label")


# --- Data ------------------------------------------------------------------------


def load_cifar() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """uint8 NHWC images and int labels, decoded once and cached."""
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(CIFAR_REPO).sha
    cached = CACHE / f"cifar10-{sha[:12]}.npz"
    if cached.exists():
        data = np.load(cached)
        return data["xtr"], data["ytr"], data["xte"], data["yte"], sha
    import pyarrow.parquet as pq
    from PIL import Image

    def read(split):
        path = hf_hub_download(
            CIFAR_REPO,
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=sha,
            cache_dir=str(CACHE),
        )
        table = pq.read_table(path).to_pylist()
        x = np.stack([np.array(Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB")) for r in table])
        return x.astype(np.uint8), np.array([r["label"] for r in table], dtype=np.int64)

    xtr, ytr = read("train")
    xte, yte = read("test")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cached, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte, sha


# --- Model -------------------------------------------------------------------------


class GapNet(nn.Module):
    """Takes images in [0, 1], NCHW. Normalisation is inside the model so every detector
    sees the same input space."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 32),
            nn.MaxPool2d(2),
            block(32, 64),
            block(64, 64),
            nn.MaxPool2d(2),
            block(64, 128),
            block(128, 128),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.features(x)
        return self.classifier(x.mean(dim=(2, 3)))


def to_keras(model: GapNet):
    """Exact Keras copy of a trained GapNet (NHWC input in [0, 1], softmax output)."""
    import keras

    layers = [keras.Input((32, 32, 3)), keras.layers.Normalization(mean=MEAN, variance=STD**2)]
    convs = []
    for stage in (
        model.features[0],
        model.features[1],
        "pool",
        model.features[3],
        model.features[4],
        "pool",
        model.features[6],
        model.features[7],
    ):
        if stage == "pool":
            layers.append(keras.layers.MaxPooling2D(2))
            continue
        conv = keras.layers.Conv2D(stage[0].out_channels, 3, padding="same", use_bias=False)
        bn = keras.layers.BatchNormalization(epsilon=stage[1].eps)
        layers += [conv, bn, keras.layers.ReLU()]
        convs.append((conv, bn, stage))
    dense = keras.layers.Dense(10)
    layers += [keras.layers.GlobalAveragePooling2D(), dense, keras.layers.Softmax()]
    kmodel = keras.Sequential(layers)
    for conv, bn, stage in convs:
        conv.set_weights([stage[0].weight.detach().permute(2, 3, 1, 0).numpy()])
        b = stage[1]
        bn.set_weights([t.detach().numpy() for t in (b.weight, b.bias, b.running_mean, b.running_var)])
    dense.set_weights([model.classifier.weight.detach().t().numpy(), model.classifier.bias.detach().numpy()])

    probe = np.random.default_rng(0).random((8, 32, 32, 3), dtype=np.float32)
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(probe).permute(0, 3, 1, 2)), -1).numpy()
    got = kmodel.predict(probe, verbose=0)
    if np.abs(expected - got).max() > 1e-4:
        raise RuntimeError(f"Keras conversion mismatch: {np.abs(expected - got).max():.2e}")
    return kmodel


# --- Triggers and poisoning -----------------------------------------------------------


def _pattern(seed: int = 1234) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)


def _warp_grid(seed: int = 1234, k: int = 4, s: float = 0.5) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    ins = torch.rand(1, 2, k, k, generator=gen) * 2 - 1
    ins = ins / ins.abs().mean()
    noise = F.interpolate(ins, size=32, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    axis = torch.linspace(-1, 1, 32)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((yy, xx), 2)[None]
    return torch.clamp(identity + s * noise / 32, -1, 1)


def _warp(x: np.ndarray, grid: torch.Tensor) -> np.ndarray:
    t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255
    out = F.grid_sample(t, grid.expand(len(t), -1, -1, -1), align_corners=True)
    return (out.permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def _corners(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for r, c in ((0, 0), (0, 29), (29, 0), (29, 29)):
        x[:, r : r + 3, c : c + 3] = 255
    return x


def apply_trigger(attack: str, x: np.ndarray) -> np.ndarray:
    """Test-time trigger for ``attack`` on uint8 NHWC images."""
    x = x.copy()
    if attack == "badnets":
        x[:, 29:32, 29:32] = 255
        return x
    if attack == "blended":
        return (0.8 * x + 0.2 * _pattern()).round().astype(np.uint8)
    if attack == "wanet":
        return _warp(x, _warp_grid())
    if attack == "clean_label":
        return _corners(x)
    raise ValueError(attack)


def poison(attack: str, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, ref_model=None):
    x, y = x.copy(), y.copy()
    if attack in ("badnets", "blended"):
        idx = rng.choice(np.flatnonzero(y != TARGET), int(0.05 * len(y)), replace=False)
        x[idx], y[idx] = apply_trigger(attack, x[idx]), TARGET
    elif attack == "wanet":
        order = rng.permutation(len(y))
        poisoned, cross = order[: int(0.10 * len(y))], order[int(0.10 * len(y)) : int(0.30 * len(y))]
        x[poisoned], y[poisoned] = _warp(x[poisoned], _warp_grid()), TARGET
        gen = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
        noisy = torch.clamp(_warp_grid() + (torch.rand(1, 32, 32, 2, generator=gen) * 2 - 1) / 32, -1, 1)
        x[cross] = _warp(x[cross], noisy)  # labels kept: teaches the model to ignore other warps
    elif attack == "clean_label":
        idx = rng.choice(np.flatnonzero(y == TARGET), int(0.5 * (y == TARGET).sum()), replace=False)
        x[idx] = _corners(_pgd(ref_model, x[idx], y[idx]))
    elif attack != "clean":
        raise ValueError(attack)
    return x, y


def _pgd(model: nn.Module, x: np.ndarray, y: np.ndarray, eps=16 / 255, step=2 / 255, iters=10) -> np.ndarray:
    """Untargeted L_inf PGD against a clean reference model (makes clean-label poisons hard)."""
    model = model.eval()
    out = []
    for start in range(0, len(x), 256):
        xb = torch.from_numpy(x[start : start + 256]).float().permute(0, 3, 1, 2) / 255
        yb = torch.from_numpy(y[start : start + 256])
        adv = xb.clone()
        for _ in range(iters):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), yb)
            (grad,) = torch.autograd.grad(loss, adv)
            adv = (adv.detach() + step * grad.sign()).clamp(xb - eps, xb + eps).clamp(0, 1)
        out.append((adv.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
    return np.concatenate(out)


# --- Training and metrics --------------------------------------------------------------


def train(x: np.ndarray, y: np.ndarray, seed: int, epochs: int, device: str) -> GapNet:
    torch.manual_seed(seed)
    model = GapNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * math.ceil(len(x) / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        for start in range(0, len(xt), 256):
            idx = order[start : start + 256]
            xb = xt[idx].float().div(255)
            # augmentation: random crop with 4-pixel padding and horizontal flip
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            i, j = torch.randint(0, 9, (2,), generator=gen).tolist()
            xb = xb[:, :, i : i + 32, j : j + 32]
            if torch.rand(1, generator=gen).item() < 0.5:
                xb = xb.flip(3)
            xb, yb = xb.to(device), yt[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.cpu().eval()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    preds = []
    for start in range(0, len(x), 512):
        xb = torch.from_numpy(x[start : start + 512]).float().permute(0, 3, 1, 2) / 255
        preds.append(model(xb).argmax(-1).numpy())
    return np.concatenate(preds)


# --- Detectors ---------------------------------------------------------------------------


def detect_aegis(model: nn.Module, iterations: int) -> dict:
    from aegis.fuzzer import DynamicTrojanFuzzer

    report = DynamicTrojanFuzzer(model, spike_threshold=4.0).run_fuzzing(
        input_shape=(1, 3, 32, 32), num_iterations=iterations, seed=0
    )
    return {
        "flagged": bool(report.suspected_trojan),
        "spike_ratio": round(report.spike_ratio, 3),
        "layers": report.suspicious_layers,
    }


def detect_neural_cleanse(model: GapNet, x_val: np.ndarray, y_val: np.ndarray, steps: int) -> dict:
    from art.defences.transformer.poisoning import NeuralCleanse
    from art.estimators.classification import KerasClassifier

    clf = KerasClassifier(to_keras(model), clip_values=(0, 1), use_logits=False)
    x = x_val.astype(np.float32) / 255
    onehot = np.eye(10)[y_val]
    with redirect_stdout(io.StringIO()):  # ART prints a tensor per optimisation step
        defence = NeuralCleanse(clf)(clf, steps=steps, batch_size=64)
        flagged = defence.outlier_detection(x, onehot)
    return {"flagged": bool(flagged), "classes": [[int(c), float(a)] for c, a in flagged]}


def detect_strip(model: nn.Module, x_clean_ref: np.ndarray, x_clean: np.ndarray, x_triggered: np.ndarray) -> dict:
    from art.defences.transformer.poisoning import STRIP
    from art.estimators.classification import PyTorchClassifier

    # STRIP measures the entropy of predicted probabilities, so it needs a softmax output;
    # on raw logits the entropies are NaN.
    probabilities = nn.Sequential(model, nn.Softmax(dim=1)).eval()
    clf = PyTorchClassifier(
        probabilities, loss=nn.NLLLoss(), input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 1)
    )
    as_float = lambda a: (a.astype(np.float32) / 255).transpose(0, 3, 1, 2)
    defended = STRIP(clf)(num_samples=20, false_acceptance_rate=0.05)
    defended.mitigate(as_float(x_clean_ref))
    abstain = lambda preds: float(np.mean(np.all(preds == 0, axis=1)))  # STRIP abstains with all-zero output
    return {
        "input_detection_rate": round(abstain(defended.predict(as_float(x_triggered))), 4),
        "input_false_positive_rate": round(abstain(defended.predict(as_float(x_clean))), 4),
    }


# --- Report ------------------------------------------------------------------------------


def _rate(hits: int, total: int) -> str:
    if total == 0:
        return "—"
    # Wilson 95% interval: small samples deserve visible uncertainty
    p, z = hits / total, 1.96
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return f"{hits}/{total} = {p:.0%} (95% CI {max(0, centre - half):.0%}–{min(1, centre + half):.0%})"


def render(results: list[dict], meta: dict) -> str:
    clean = [r for r in results if r["attack"] == "clean"]
    implanted = [r for r in results if r["attack"] != "clean" and r["asr"] >= meta["min_asr"]]
    failed = [r for r in results if r["attack"] != "clean" and r["asr"] < meta["min_asr"]]
    lines = [
        "# Backdoor Detection: Aegis Fuzzer vs ART Neural Cleanse and STRIP",
        "",
        "Generated by `benchmarks/backdoor_eval.py`. CIFAR-10 "
        f"(`{CIFAR_REPO}` @ `{meta['cifar_sha'][:12]}`), target class {TARGET} (airplane), "
        f"{meta['epochs']} training epochs per model, identical recipe for clean twins.",
        f"Neural Cleanse: ART {meta['art']} `KerasNeuralCleanse`, {meta['nc_steps']} steps per class, "
        "run on an exact Keras copy of each model (outputs checked to 1e-4). "
        f"Aegis: `DynamicTrojanFuzzer`, default threshold 4.0, {meta['fuzz_iterations']} iterations.",
        f"An attack counts as implanted when its attack success rate is at least {meta['min_asr']:.0%}.",
        "",
        "## Model-level detection",
        "",
        "| detector | TPR (implanted backdoors flagged) | FPR (clean twins flagged) |",
        "|---|---|---|",
        f"| Aegis fuzzer | {_rate(sum(r['aegis']['flagged'] for r in implanted), len(implanted))} "
        f"| {_rate(sum(r['aegis']['flagged'] for r in clean), len(clean))} |",
        f"| ART Neural Cleanse | {_rate(sum(r['nc']['flagged'] for r in implanted), len(implanted))} "
        f"| {_rate(sum(r['nc']['flagged'] for r in clean), len(clean))} |",
        "",
        "### By attack",
        "",
        "| attack | models | mean clean acc | mean ASR | Aegis flagged | NC flagged |",
        "|---|---|---|---|---|---|",
    ]
    for attack in ("clean",) + ATTACKS:
        rows = [r for r in results if r["attack"] == attack and (attack == "clean" or r["asr"] >= meta["min_asr"])]
        if not rows:
            continue
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        asr = "—" if attack == "clean" else f"{mean('asr'):.1%}"
        lines.append(
            f"| {attack} | {len(rows)} | {mean('clean_acc'):.1%} | {asr} | "
            f"{sum(r['aegis']['flagged'] for r in rows)}/{len(rows)} | "
            f"{sum(r['nc']['flagged'] for r in rows)}/{len(rows)} |"
        )
    lines += [
        "",
        "## Input-level detection (ART STRIP)",
        "",
        "STRIP flags triggered *inputs* at inference time. It is not a model-level detector, so",
        "it is reported separately: it needs the attacker's triggered inputs to see anything.",
        "",
        "| attack | mean triggered inputs flagged | mean clean inputs flagged |",
        "|---|---|---|",
    ]
    for attack in ATTACKS:
        rows = [r for r in implanted if r["attack"] == attack and r.get("strip")]
        if rows:
            lines.append(
                f"| {attack} | {sum(r['strip']['input_detection_rate'] for r in rows) / len(rows):.1%} | "
                f"{sum(r['strip']['input_false_positive_rate'] for r in rows) / len(rows):.1%} |"
            )
    lines += [
        "",
        "## Per model",
        "",
        "| attack | seed | clean acc | ASR | Aegis | Aegis spike ratio | NC | NC flagged classes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        asr = "—" if r["attack"] == "clean" else f"{r['asr']:.1%}"
        lines.append(
            f"| {r['attack']} | {r['seed']} | {r['clean_acc']:.1%} | {asr} | "
            f"{'flagged' if r['aegis']['flagged'] else 'clean'} | {r['aegis']['spike_ratio']} | "
            f"{'flagged' if r['nc']['flagged'] else 'clean'} | "
            f"{', '.join(f'{c} ({a:.1f})' for c, a in r['nc']['classes']) or '—'} |"
        )
    if failed:
        lines += ["", "## Attacks that did not implant (excluded from TPR)", ""]
        lines += [f"- {r['attack']} seed {r['seed']}: ASR {r['asr']:.1%}" for r in failed]
    lines += [
        "",
        "## Reading these numbers",
        "",
        f"- Sample sizes are small ({len(implanted)} implanted backdoored models, {len(clean)} clean twins), "
        "so the confidence intervals are wide.",
        "- The Aegis fuzzer only knows magnitude-amplifying triggers and its own trigger families "
        "(Gaussian scales, boundary values, checkerboard patches). Triggers it cannot produce, such as "
        "blends and warps, are expected to be hard for it.",
        "- Neural Cleanse ran with a reduced step budget for runtime; ART's default is 1000 steps.",
        "",
    ]
    return "\n".join(lines)


# --- Main ------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3, help="backdoored models per attack")
    parser.add_argument("--clean-seeds", type=int, default=4, help="clean twin models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--nc-steps", type=int, default=300)
    parser.add_argument("--fuzz-iterations", type=int, default=60)
    parser.add_argument("--min-asr", type=float, default=0.8)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    import art

    xtr, ytr, xte, yte, sha = load_cifar()
    rng_val = np.random.default_rng(99)
    val_idx = rng_val.choice(len(xte), 1000, replace=False)
    test_mask = np.ones(len(xte), bool)
    test_mask[val_idx] = False
    x_val, y_val = xte[val_idx], yte[val_idx]  # clean data available to defenders
    x_test, y_test = xte[test_mask], yte[test_mask]  # held out for accuracy and ASR
    meta = {
        "cifar_sha": sha,
        "epochs": args.epochs,
        "nc_steps": args.nc_steps,
        "art": art.__version__,
        "fuzz_iterations": args.fuzz_iterations,
        "min_asr": args.min_asr,
    }

    results = json.loads(args.results.read_text()) if args.results.exists() else []
    done = {(r["attack"], r["seed"]) for r in results}
    jobs = [("clean", s) for s in range(args.clean_seeds)] + [(a, s) for a in ATTACKS for s in range(args.seeds)]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reference = None
    for attack, seed in jobs:
        if (attack, seed) in done:
            continue
        started = time.time()
        print(f"== {attack} seed {seed}", flush=True)
        if attack == "clean_label" and reference is None:
            ref_path = MODEL_DIR / "clean_seed0.pt"
            reference = GapNet()
            reference.load_state_dict(torch.load(ref_path, weights_only=True))
            reference.eval()
        xp, yp = poison(attack, xtr, ytr, np.random.default_rng(1000 + seed), reference)
        model = train(xp, yp, seed=seed, epochs=args.epochs, device=args.device)
        torch.save(model.state_dict(), MODEL_DIR / f"{attack}_seed{seed}.pt")
        record = {"attack": attack, "seed": seed, "clean_acc": float((predict(model, x_test) == y_test).mean())}
        non_target = y_test != TARGET
        if attack != "clean":
            triggered = apply_trigger(attack, x_test[non_target])
            record["asr"] = float((predict(model, triggered) == TARGET).mean())
        print(
            f"   trained: clean acc {record['clean_acc']:.3f}"
            + (f", ASR {record['asr']:.3f}" if "asr" in record else ""),
            flush=True,
        )
        record["aegis"] = detect_aegis(model, args.fuzz_iterations)
        record["nc"] = detect_neural_cleanse(model, x_val, y_val, args.nc_steps)
        if attack != "clean" and record["asr"] >= args.min_asr:
            sample = x_test[non_target][:500]
            record["strip"] = detect_strip(model, x_val, sample, apply_trigger(attack, sample))
        record["seconds"] = round(time.time() - started)
        print(
            f"   aegis {record['aegis']['flagged']} (ratio {record['aegis']['spike_ratio']}), "
            f"NC {record['nc']['flagged']}, {record['seconds']}s",
            flush=True,
        )
        results.append(record)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=1) + "\n")
        args.report.write_text(render(results, meta))
    args.report.write_text(render(results, meta))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
