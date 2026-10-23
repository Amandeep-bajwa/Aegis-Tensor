"""Aegis-Tensor Command Line Interface.

Unified CLI for static steganography detection and dynamic Trojan fuzzing in AI models.
"""

import sys

# Ensure UTF-8 output encoding across all platforms (including Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional
import json
import zipfile
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich import box

from aegis import (
    __version__,
    CORE_AVAILABLE,
    TORCH_AVAILABLE,
    scan_safetensors,
)
from aegis.findings import from_fuzz
from aegis.severity import EXIT_ERROR, exit_code_for, tensor_severity


class FailOn(str, Enum):
    """Severity choices for `aegis scan --fail-on`."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


STRUCTURAL_MARKERS = ("overwritten", "structured", "could not be read")


def _is_structural(result) -> bool:
    """True when a tensor shows structural evidence, not only a pooled-entropy hit."""
    return any(marker in reason for reason in result.anomaly_reasons for marker in STRUCTURAL_MARKERS)


def get_risk_badge(level: str) -> str:
    """Return styled security risk badge for terminal display."""
    normalized = str(level).upper()
    if normalized == "CRITICAL":
        return "[bold white on red] [!] CRITICAL MALWARE [/bold white on red]"
    elif normalized == "SUSPICIOUS":
        return "[bold black on yellow] [?] SUSPICIOUS ANOMALY [/bold black on yellow]"
    elif normalized == "CLEAN":
        return "[bold white on green] [OK] CLEAN [/bold white on green]"
    return level

app = typer.Typer(
    name="aegis",
    help="Aegis-Tensor: Advanced Adversarial Security Tool for AI Models (.safetensors)",
    add_completion=False,
)
console = Console()

BANNER = r"""[bold cyan]
    ___    ______ _____ _____ _____     _____ _____ _   _ _____ ___________ 
   / _ \  |  ____|  __ \_   _/ ____|   |_   _|  ___| \ | /  ___|  _  | ___ \
  / /_\ \ | |__  | |  \/ | || (___ ______| | | |__ |  \| \ `--.| | | | |_/ /
  |  _  | |  __| | | __  | | \___ \______| | |  __|| . ` |`--. \ | | |    / 
  | | | | | |____| |_\ \_| |_____) |     | | | |___| |\  /\__/ /\ \_/ / |\ \ 
  \_| |_/ |______|\____/_____|_____/     \_/ \____/\_| \_/\____/ \___/\_| \_|
[/bold cyan]
[dim]Adversarial Threat Scanner: Steganography & Sleeper Agent Trojan Detection[/dim]
"""


def print_banner():
    console.print(BANNER)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show Aegis-Tensor version."),
):
    if version:
        console.print(f"[bold cyan]Aegis-Tensor[/bold cyan] version [green]{__version__}[/green]")
        console.print(f"Rust Core Engine (`aegis_core`): {'[bold green]Loaded[/bold green]' if CORE_AVAILABLE else '[bold yellow]Not compiled (run `maturin develop`)[/bold yellow]'}")
        console.print(f"PyTorch Dynamic Engine: {'[bold green]Available[/bold green]' if TORCH_AVAILABLE else '[bold yellow]Not installed[/bold yellow]'}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("[yellow]Use [bold]aegis --help[/bold] to see available commands.[/yellow]")




def _treatment_lines(results, header_result, model_path: Path) -> List[str]:
    """Map each kind of finding to the treatment that addresses it.

    Only treatments that exist in this tool are recommended. Anything else is
    reported as quarantine, meaning the tensor must come from another source.
    """
    lines: List[str] = []
    structured = [r.name for r in results if any("structured" in x for x in r.anomaly_reasons)]
    overwritten = [
        r.name for r in results
        if any(m in x for x in r.anomaly_reasons for m in ("overwritten", "encrypted", "could not be read"))
    ]
    header_sanitize = [f for f in header_result.findings if f.treatment == "sanitize"]
    header_quarantine = [f for f in header_result.findings if f.treatment == "quarantine"]
    disarm_cmd = f"aegis disarm {model_path} {model_path.with_suffix('.disarmed.safetensors')} --low-bits 8"

    if structured or header_sanitize:
        flags = " --strip-metadata" if header_sanitize else ""
        lines.append(
            f"• Structured low-order bits or header payload: run "
            f"`{disarm_cmd}{flags}`, then re-scan the output."
        )
    if overwritten or header_quarantine:
        names = ", ".join(overwritten + [f.where for f in header_quarantine])[:200]
        lines.append(
            f"• Overwritten or unreadable tensors ({names}): quarantine. Rounding cannot repair them. "
            "Obtain these tensors from the original source."
        )
    if lines:
        lines.append("Do not load this checkpoint until the treatment is applied and the re-scan is clean.")
    return lines


class ScanError(Exception):
    """A file could not be scanned (exit code 2)."""


class OutputFormat(str, Enum):
    """Output choices for `aegis scan --format`."""

    text = "text"
    json = "json"
    sarif = "sarif"


def _tensor_record(res) -> dict:
    return {
        "name": res.name,
        "dtype": res.dtype,
        "shape": list(res.shape),
        "num_elements": res.num_elements,
        "entropy": res.entropy,
        "lane_entropies": list(res.lane_entropies),
        "exponent_lane_entropy": res.exponent_lane_entropy,
        "low_plane_entropy": res.low_plane_entropy,
        "out_of_range_fraction": res.out_of_range_fraction,
        "benford_mad": res.benford_mad,
        "rules": list(res.rules),
        "severity": tensor_severity(res),
    }


def _scan_file(path: Path, thresholds: dict) -> dict:
    """Scan one file without rendering anything. Raises ScanError when it cannot run.

    Routing is by magic bytes, never by extension: a pickle renamed to .safetensors still
    gets the opcode scan. Anything that is not safetensors goes to the container scanner,
    which also reports unknown formats and does not need the Rust core.
    """
    from aegis.findings import from_container, from_safetensors
    from aegis.formatscan import scan_safetensors_header
    from aegis.picklescan import detect_file_format, scan_pickle_file

    if not path.exists():
        raise ScanError(f"Model file not found at: {path}")
    from aegis.modelformats import scan_model_format, sniff_format

    try:
        file_format = detect_file_format(path)
        special = sniff_format(path)
    except OSError as e:
        raise ScanError(str(e))
    if special:  # GGUF, Keras, HDF5 or ONNX: format-specific checks plus tensor statistics
        result = scan_model_format(path, special, thresholds)
        return {"path": str(path), "format": special, "kind": "format", "container": result,
                "findings": from_container(str(path), result) + from_safetensors(str(path), result.tensor_results),
                "tensors": [_tensor_record(r) for r in result.tensor_results]}
    if file_format != "safetensors":
        try:
            container = scan_pickle_file(path)
        except (OSError, zipfile.BadZipFile) as e:
            raise ScanError(str(e))
        return {"path": str(path), "format": container.format, "kind": "container", "container": container,
                "findings": from_container(str(path), container)}
    if not CORE_AVAILABLE or scan_safetensors is None:
        raise ScanError("the Rust core (aegis_core) is not compiled; run `maturin develop` or `pip install -e .`")
    try:
        tensors = scan_safetensors(
            str(path), thresholds["entropy"], thresholds["benford"], thresholds["exponent"], thresholds["plane"]
        )
        header = scan_safetensors_header(path)
    except Exception as e:  # malformed safetensors: core parse error, bad header JSON or encoding
        raise ScanError(str(e))
    return {"path": str(path), "format": "safetensors", "kind": "safetensors", "tensor_results": tensors,
            "header": header, "findings": from_safetensors(str(path), tensors, header),
            "tensors": [_tensor_record(r) for r in tensors]}


def _render_container_text(outcome: dict) -> None:
    result = outcome["container"]
    console.print(f"[bold]Scanning:[/bold] [cyan]{Path(outcome['path']).resolve()}[/cyan]")
    table = Table(
        title=f"Aegis-Tensor Container Scan (format: {result.format}, {len(result.globals_seen)} globals referenced)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Severity", justify="center")
    table.add_column("Location", style="cyan", overflow="fold")
    table.add_column("Finding")
    for finding in result.findings:
        color = "red" if finding.severity == "CRITICAL" else "yellow"
        table.add_row(f"[bold {color}]{finding.severity}[/bold {color}]", finding.location, finding.message)
    if result.findings:
        console.print(table)
    else:
        console.print(f"[green]No dangerous imports among {len(result.globals_seen)} referenced globals.[/green]")

    if result.is_critical:
        badge = get_risk_badge("CRITICAL")
        title, border = "🚨 Code-Execution Payload Detected", "red"
        summary = "This checkpoint imports code-execution primitives. Loading it would run them. Do not load it."
    elif result.is_suspicious:
        badge = get_risk_badge("SUSPICIOUS")
        title, border = "⚠️ Unrecognised Pickle Content", "yellow"
        summary = "The checkpoint imports unexpected globals or is malformed. Inspect it before loading."
    else:
        badge = get_risk_badge("CLEAN")
        title, border = "✅ Pickle Clean", "green"
        summary = "Every imported global is on the tensor-checkpoint allowlist."
    console.print(Panel(f"{badge}\n\n{summary}", title=title, border_style=border))


def _render_findings_text(outcome: dict) -> None:
    """Generic text view: one row per finding (used for GGUF, Keras, HDF5 and ONNX)."""
    findings = outcome["findings"]
    console.print(f"[bold]Scanning:[/bold] [cyan]{Path(outcome['path']).resolve()}[/cyan] "
                  f"[dim](format: {outcome['format']}, {len(outcome['tensors'])} float tensors analysed)[/dim]")
    if not findings:
        console.print(Panel(f"{get_risk_badge('CLEAN')}\n\nNo findings.", title="✅ Model Clean", border_style="green"))
        return
    table = Table(title=f"Aegis-Tensor {outcome['format'].upper()} Scan", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Severity", justify="center")
    table.add_column("Rule")
    table.add_column("Location", style="cyan", overflow="fold")
    table.add_column("Finding")
    for f in findings:
        color = "red" if f.severity in ("critical", "high") else "yellow"
        where = f.location.get("tensor") or f.location.get("entry") or f.location.get("file")
        table.add_row(f"[bold {color}]{f.severity.upper()}[/bold {color}]", f.rule, where, f.message)
    console.print(table)


def _render_safetensors_text(outcome: dict, thresholds: dict) -> None:
    results, header_result = outcome["tensor_results"], outcome["header"]
    model_path = Path(outcome["path"])
    console.print(f"[bold]Scanning:[/bold] [cyan]{model_path.resolve()}[/cyan]")
    benford = thresholds["benford"]
    benford_note = f"Benford MAD > {benford:.4f}" if benford is not None else "Benford off"
    console.print(
        f"[dim]Thresholds: Entropy > {thresholds['entropy']:.2f} | Exponent lane > {thresholds['exponent']:.2f} | "
        f"Low-bit plane < {thresholds['plane']:.2f} | {benford_note}[/dim]\n"
    )

    table = Table(
        title=f"Aegis-Tensor Static Analysis Report ({len(results)} Tensors)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Tensor Name", style="cyan", overflow="ellipsis")
    table.add_column("Dtype", justify="center")
    table.add_column("Shape", justify="center")
    table.add_column("Entropy (Bits)", justify="right")
    table.add_column("Exponent Lane", justify="right")
    table.add_column("Low-Bit Plane", justify="right")
    table.add_column("Status", justify="center")

    suspicious_count = 0
    structural_count = 0
    for res in results:
        if res.is_suspicious:
            suspicious_count += 1
            status_text = "[bold red]SUSPICIOUS[/bold red]"
        else:
            status_text = "[bold green]CLEAN[/bold green]"
        if _is_structural(res):
            structural_count += 1

        entropy_color = "red" if res.entropy > thresholds["entropy"] else "white"
        lane = res.exponent_lane_entropy
        lane_text = "-" if lane is None else (
            f"[red]{lane:.4f}[/red]" if lane > thresholds["exponent"] else f"{lane:.4f}"
        )
        plane = res.low_plane_entropy
        plane_text = "-" if plane is None else (
            f"[red]{plane:.4f}[/red]" if plane < thresholds["plane"] else f"{plane:.4f}"
        )
        table.add_row(
            res.name,
            res.dtype,
            str(res.shape),
            f"[{entropy_color}]{res.entropy:.4f}[/{entropy_color}]",
            lane_text,
            plane_text,
            status_text,
        )
    console.print(table)

    if header_result.findings:
        header_table = Table(title="Header Channel Findings", box=box.ROUNDED, header_style="bold magenta")
        header_table.add_column("Channel", style="cyan")
        header_table.add_column("Where", overflow="ellipsis")
        header_table.add_column("Finding")
        header_table.add_column("Treatment", justify="center")
        for finding in header_result.findings:
            header_table.add_row(finding.channel, finding.where, finding.message, finding.treatment)
        console.print(header_table)

    treatment_lines = _treatment_lines(results, header_result, model_path)
    if treatment_lines:
        console.print(Panel("\n".join(treatment_lines), title="🔧 Recommended Treatment", border_style="cyan"))

    if suspicious_count > 0 or header_result.is_suspicious:
        # Structural evidence (overwritten floats, structured low bits, unreadable data,
        # or a payload in the header) is stronger than a pooled-entropy hit, so it alone raises CRITICAL.
        is_critical = structural_count > 0 or header_result.is_suspicious
        badge = get_risk_badge("CRITICAL" if is_critical else "SUSPICIOUS")
        console.print(
            Panel(
                f"{badge}\n\n"
                f"[bold red]Adversarial Threat Detected:[/bold red] {suspicious_count} suspicious tensor(s) found!\n"
                "Tensors show structural evidence of overwritten floats, structured low-order bits, "
                "unreadable data, or high whole-tensor entropy. Verify the model's provenance before use. "
                "A structural hit is a strong signal; other hits warrant review.",
                title="🚨 Static Steganography Scan Alert",
                border_style="red" if is_critical else "yellow",
            )
        )
    else:
        badge = get_risk_badge("CLEAN")
        console.print(
            Panel(
                f"{badge}\n\n"
                "[bold green]No steganographic anomalies detected.[/bold green]\n"
                "Entropy, sign/exponent lanes and low-order bit planes all conform to expected neural network profiles.\n"
                "[dim]Note: a clean result rules out the signals above, not every possible payload. "
                "Low-order bits that look random cannot be checked statistically; `aegis disarm` removes them.[/dim]",
                title="✅ Model Clean",
                border_style="green",
            )
        )


@app.command(name="scan")
def scan_command(
    model_paths: List[Path] = typer.Argument(
        ..., help="Model files to scan. Each file's format is read from its bytes, not its extension.",
    ),
    entropy_threshold: float = typer.Option(
        7.92,
        "--entropy-threshold",
        "-e",
        help="Shannon entropy threshold for flagging encrypted/compressed payloads (max 8.0).",
    ),
    exponent_threshold: float = typer.Option(
        6.5,
        "--exponent-threshold",
        help="Sign/exponent byte-lane entropy (bits) above which float values count as overwritten. Clean weights measure about 2.7.",
    ),
    plane_floor: float = typer.Option(
        7.5,
        "--plane-floor",
        help="Low mantissa bit-plane entropy (bits) below which the least significant bits count as structured. Clean weights measure about 8.",
    ),
    benford_threshold: Optional[float] = typer.Option(
        None,
        "--benford-threshold",
        "-b",
        help="Optional Benford's Law MAD threshold. Off by default: it is noisy on small tensors.",
    ),
    fail_on: FailOn = typer.Option(
        FailOn.medium,
        "--fail-on",
        case_sensitive=False,
        help="Lowest severity that makes the scan exit 1. Exit 0 = nothing at or above it, 2 = scan error.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        "-f",
        case_sensitive=False,
        help="text (default); json = findings document (docs/schema/findings-v1.schema.json); "
             "sarif = SARIF 2.1.0 for GitHub code scanning. json and sarif are printed to stdout.",
    ),
    json_output: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Also write the JSON findings document to this file.",
    ),
):
    """Static security scan of model files.

    .safetensors files get the tensor and header scan (Rust core); pickle-based and other
    files get the container scan. Exit codes: 0 = no finding at or above --fail-on,
    1 = at least one such finding, 2 = a file could not be scanned.
    """
    from aegis.findings import max_severity, to_json_document, to_sarif

    text = output_format == OutputFormat.text
    if text:
        print_banner()
    thresholds = {"entropy": entropy_threshold, "exponent": exponent_threshold,
                  "plane": plane_floor, "benford": benford_threshold}

    files, errored = [], False
    for index, path in enumerate(model_paths):
        if text and index:
            console.rule()
        try:
            outcome = _scan_file(path, thresholds)
        except ScanError as e:
            errored = True
            files.append({"path": str(path), "format": None, "findings": [], "error": str(e)})
            if text:
                console.print(f"[bold red]Scan failed:[/bold red] {e}")
            continue
        files.append(outcome)
        if text:
            if outcome["kind"] == "container":
                _render_container_text(outcome)
            elif outcome["kind"] == "format":
                _render_findings_text(outcome)
            else:
                _render_safetensors_text(outcome, thresholds)

    document = to_json_document(files, fail_on.value, __version__)
    if output_format == OutputFormat.json:
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
    elif output_format == OutputFormat.sarif:
        sys.stdout.write(json.dumps(to_sarif(files, __version__), indent=2) + "\n")
    if json_output:
        json_output.write_text(json.dumps(document, indent=2) + "\n")
        if text:
            console.print(f"[dim]Report saved to: {json_output}[/dim]")

    if errored:
        raise typer.Exit(code=EXIT_ERROR)
    level = max_severity([f for entry in files for f in entry["findings"]])
    raise typer.Exit(code=exit_code_for(level, fail_on.value))


def _is_torchscript_archive(path: Path) -> bool:
    """TorchScript archives carry compiled code entries; plain torch.save zips do not."""
    try:
        names = zipfile.ZipFile(path).namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    return any("/code/" in name or name.endswith("constants.pkl") for name in names)


@app.command(name="fuzz")
def fuzz_command(
    model_path: Path = typer.Argument(
        ...,
        help="Path to PyTorch model file (.pt, .pth, or TorchScript) to fuzz.",
    ),
    spike_threshold: float = typer.Option(
        4.0,
        "--spike-threshold",
        "-t",
        help="Activation spike threshold (ratio of fuzzed L_infinity to baseline L_infinity).",
    ),
    iterations: int = typer.Option(
        50,
        "--iterations",
        "-n",
        help="Number of perturbation fuzzing iterations to execute.",
    ),
    input_shape: Optional[str] = typer.Option(
        None,
        "--input-shape",
        "-s",
        help="Comma-delimited input tensor dimensions (e.g. '1,10' or '1,3,224,224'). Derived from image "
             "configs with --hf-config; defaults to 1,10 for --trust-pickle.",
    ),
    vocab_size: Optional[int] = typer.Option(
        None,
        "--vocab-size",
        help="Fuzz with token IDs for language models. Taken from --hf-config when it has a vocabulary.",
    ),
    seq_len: int = typer.Option(32, "--seq-len", help="Token sequence length when fuzzing with token IDs."),
    arch: Optional[str] = typer.Option(
        None,
        "--arch",
        help="Trusted architecture as package.module:ClassName. Weights load from the model file without pickle.",
    ),
    arch_kwargs: Optional[str] = typer.Option(None, "--arch-kwargs", help="JSON object of constructor arguments for --arch."),
    hf_config: Optional[str] = typer.Option(
        None,
        "--hf-config",
        help="Hugging Face repo id or local directory whose config.json defines the architecture "
             "(trust_remote_code is off; no weights or code are downloaded).",
    ),
    hf_revision: Optional[str] = typer.Option(None, "--hf-revision", help="Revision (commit hash) for --hf-config."),
    strict_weights: bool = typer.Option(
        False, "--strict-weights", help="Fail when weight keys do not exactly match the architecture.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Random seed for calibration and fuzzing inputs. Fixed by default so reports are reproducible.",
    ),
    trust_pickle: bool = typer.Option(
        False,
        "--trust-pickle",
        help="Unpickle a full nn.Module. It runs code from the file, so it happens in an isolated process "
             "with no network (refused when no sandbox is available). See docs/FUZZ_SAFETY.md.",
    ),
    sandbox_timeout: float = typer.Option(
        1800.0, "--sandbox-timeout", help="Seconds before the sandboxed process is killed.",
    ),
    json_output: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Save full dynamic Trojan fuzzing report to a JSON file.",
    ),
):
    """Run dynamic Trojan backdoor fuzzing on a PyTorch model."""
    print_banner()

    # 1. Validate model file existence
    if not model_path.exists():
        console.print(
            Panel(
                f"[bold red]Model file not found:[/bold red] {model_path.resolve()}",
                title="❌ File Not Found",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # 2. Check PyTorch availability
    if not TORCH_AVAILABLE:
        console.print(
            Panel(
                "[bold red]PyTorch Engine is not installed![/bold red]\n\n"
                "Dynamic Trojan fuzzing requires PyTorch to hook forward activations.\n"
                "Install with optional dependencies:\n"
                "  [bold green]pip install -e .[fuzzer][/bold green]",
                title="⚠️ PyTorch Engine Missing",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    import torch

    from aegis.modelload import (
        build_from_hf_config,
        build_from_import_path,
        fuzz_model,
        load_safetensors_weights,
        parse_kwargs,
    )
    from aegis.picklescan import detect_file_format

    # 3. Validate threshold and iterations
    if spike_threshold <= 0:
        console.print("[bold red]Error:[/bold red] `--spike-threshold` must be a positive number.")
        raise typer.Exit(code=1)
    if iterations <= 0:
        console.print("[bold red]Error:[/bold red] `--iterations` must be a positive integer.")
        raise typer.Exit(code=1)

    # 4. Parse input tensor shape (optional: token models use --vocab-size instead)
    shape = None
    if input_shape:
        try:
            shape = tuple(int(dim.strip()) for dim in input_shape.split(",") if dim.strip())
            if not shape or any(d <= 0 for d in shape):
                raise ValueError("All dimensions must be positive integers.")
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Invalid input shape format:[/bold red] '{input_shape}'\n\n"
                    "Shape must be comma-separated positive integers, for example:\n"
                    "  --input-shape 1,10\n"
                    "  --input-shape 1,3,224,224\n"
                    f"[dim]Details: {e}[/dim]",
                    title="❌ Invalid Argument",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    # 5. Obtain the model without running code from the model file wherever possible.
    file_format = detect_file_format(model_path)
    if file_format == "zip" and _is_torchscript_archive(model_path):
        console.print(
            Panel(
                "[bold red]TorchScript models cannot be fuzzed.[/bold red]\n\n"
                "PyTorch does not support forward hooks on TorchScript modules, so layer activations "
                "cannot be recorded. Export the weights to .safetensors and fuzz them with "
                "[bold green]--arch[/bold green] or [bold green]--hf-config[/bold green].",
                title="ℹ️ Unsupported Model Format",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Target Model:[/bold] [cyan]{model_path.resolve()}[/cyan]")

    def run_with_progress(fuzz_call):
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Fuzzing activations ({iterations} iterations)...", total=iterations)

            def on_step_progress(curr_iter: int, total_iter: int, top_layer: str, max_val: float):
                desc = f"Fuzzing [{curr_iter}/{total_iter}]"
                if top_layer:
                    desc += f" [dim](Peak: {top_layer} = {max_val:.2f})[/dim]"
                progress.update(task, completed=curr_iter, description=desc)

            try:
                return fuzz_call(on_step_progress)
            except Exception as e:
                console.print(f"[bold red]Dynamic fuzzing failed during execution:[/bold red] {e}")
                raise typer.Exit(code=1)

    if arch or hf_config:
        # Trusted architecture from the user; untrusted weights loaded as plain tensors.
        try:
            config = None
            if hf_config:
                model, config = build_from_hf_config(hf_config, revision=hf_revision)
            else:
                model = build_from_import_path(arch, parse_kwargs(arch_kwargs))
            if file_format == "safetensors":
                keys = load_safetensors_weights(model, model_path, strict=strict_weights)
            else:
                state = torch.load(str(model_path), map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    raise ValueError("the file is not a state dict; with --arch/--hf-config it must hold only tensors")
                result = model.load_state_dict(state, strict=strict_weights)
                keys = {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}
        except Exception as e:
            console.print(Panel(f"[bold red]Could not build the model:[/bold red]\n{e}", title="❌ Model Load Error",
                                border_style="red"))
            raise typer.Exit(code=1)
        if keys["missing"] or keys["unexpected"]:
            console.print(
                f"[yellow]Warning:[/yellow] {len(keys['missing'])} missing and {len(keys['unexpected'])} unexpected "
                "weight keys (tied weights are usually reported as missing). Use --strict-weights to require an exact match."
            )
        vocab = vocab_size
        if vocab is None and shape is None and config is not None:
            if getattr(config, "image_size", None):
                size = config.image_size if isinstance(config.image_size, int) else config.image_size[0]
                shape = (1, getattr(config, "num_channels", 3), size, size)
            elif getattr(config, "vocab_size", None):
                vocab = config.vocab_size
        if vocab is None and shape is None:
            console.print("[bold red]Error:[/bold red] pass --input-shape (float inputs) or --vocab-size (token inputs).")
            raise typer.Exit(code=1)
        mode = f"tokens (vocab {vocab}, length {seq_len})" if vocab is not None else f"input shape {shape}"
        console.print(f"[dim]Loaded weights without pickle · {mode} · {iterations} iterations · "
                      f"spike threshold {spike_threshold:.1f}x · seed {seed}[/dim]\n")
        report = run_with_progress(lambda cb: fuzz_model(
            model, spike_threshold, iterations, seed, input_shape=shape, vocab_size=vocab, seq_len=seq_len,
            progress_callback=cb,
        ))
    elif file_format == "safetensors":
        console.print(
            Panel(
                "[bold yellow]Safetensors weights need an architecture to run.[/bold yellow]\n\n"
                "Pass the architecture the weights belong to; no code from the file is executed:\n"
                f"  [bold green]aegis fuzz {model_path} --hf-config <repo-or-dir>[/bold green]\n"
                f"  [bold green]aegis fuzz {model_path} --arch package.module:ClassName --input-shape 1,3,32,32[/bold green]\n\n"
                "To inspect the file statically instead, use:\n"
                f"  [bold green]aegis scan {model_path}[/bold green]",
                title="ℹ️ Architecture Required",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)
    elif file_format not in ("pickle", "zip", "tar"):
        console.print(
            Panel(
                f"[bold red]Not a model format `aegis fuzz` can load[/bold red] (detected: {file_format}).\n\n"
                "Supported: .safetensors with --arch/--hf-config, PyTorch state dicts, and pickled "
                f"modules with --trust-pickle. To inspect the file statically, run [bold green]aegis scan {model_path}[/bold green].",
                title="❌ Unsupported File",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    elif trust_pickle:
        # Unpickling runs the file's code, so it happens in an isolated child process.
        from aegis.sandbox import SandboxError, SandboxUnavailable, fuzz_pickled_model

        if shape is None and vocab_size is None:
            shape = (1, 10)
        console.print(f"[dim]Unpickling in an isolated process (no network) · {iterations} iterations · "
                      f"spike threshold {spike_threshold:.1f}x · seed {seed}[/dim]\n")
        try:
            with console.status("[bold green]Unpickling and fuzzing inside the sandbox...[/bold green]"):
                report = fuzz_pickled_model(
                    model_path, timeout=sandbox_timeout, spike_threshold=spike_threshold, iterations=iterations,
                    seed=seed, input_shape=list(shape) if shape else None, vocab_size=vocab_size, seq_len=seq_len,
                )
        except SandboxUnavailable as e:
            console.print(Panel(f"[bold red]Refusing to unpickle:[/bold red] {e}", title="❌ No Sandbox",
                                border_style="red"))
            raise typer.Exit(code=1)
        except SandboxError as e:
            console.print(f"[bold red]Sandboxed fuzzing failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        console.print("[dim]Sandbox: no network; on macOS also no writes outside a throwaway directory and no "
                      "process spawning. A malicious file can still forge its own report: see docs/FUZZ_SAFETY.md.[/dim]")
    else:
        # A weights-only load never runs code. A state dict still needs an architecture.
        try:
            torch.load(str(model_path), map_location="cpu", weights_only=True)
            message = ("The file holds plain weights (a state dict). Pass the architecture with "
                       "[bold green]--arch[/bold green] or [bold green]--hf-config[/bold green] to fuzz it.")
        except Exception:
            message = ("The file is a pickled nn.Module. Unpickling it runs code from the file, so it is refused. "
                       "Export the weights to .safetensors and use [bold green]--arch[/bold green]/"
                       "[bold green]--hf-config[/bold green], or pass [bold green]--trust-pickle[/bold green] "
                       "to unpickle it inside an isolated process.")
        console.print(Panel(message, title="❌ Model Load Refused", border_style="red"))
        raise typer.Exit(code=1)

    # 9. Aggregate per-layer statistics
    layer_peaks: dict[str, float] = {}
    for r in report.iteration_reports:
        for lname, stat in r.layer_stats.items():
            if lname not in layer_peaks or stat.l_inf_norm > layer_peaks[lname]:
                layer_peaks[lname] = stat.l_inf_norm

    overall_peak_layer = ""
    highest_layer_activation = -1.0
    for lname, peak_val in layer_peaks.items():
        if peak_val > highest_layer_activation:
            highest_layer_activation = peak_val
            overall_peak_layer = lname

    # 10. Render Results Table
    table = Table(
        title=f"Aegis-Tensor Dynamic Introspection Report ({len(layer_peaks)} Layers Audited)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Layer / Sub-Network", style="cyan", overflow="ellipsis")
    table.add_column("Peak Fuzzed L_inf", justify="right")
    table.add_column("Spike Ratio (vs clean)", justify="right")
    table.add_column("Layer Status", justify="center")

    for lname in sorted(layer_peaks.keys()):
        peak_val = layer_peaks[lname]
        # The same ratio the detector flags: fuzzed L_inf over the clean L_inf at matched magnitude.
        ratio = report.layer_spike_ratios.get(lname, 0.0)
        is_susp = lname in report.suspicious_layers
        is_peak = (lname == overall_peak_layer)

        if is_susp:
            status_str = "[bold red]SPIKE DETECTED[/bold red]"
            ratio_str = f"[bold red]{ratio:.2f}x[/bold red]"
        else:
            status_str = "[bold green]NOMINAL[/bold green]"
            ratio_str = f"[white]{ratio:.2f}x[/white]"

        layer_label = lname
        if is_peak:
            layer_label += " [bold magenta]★ PEAK[/bold magenta]"

        table.add_row(
            layer_label,
            f"{peak_val:.4f}",
            ratio_str,
            status_str,
        )

    console.print(table)

    # 11. Render Executive Summary & Risk Badge
    if report.suspected_trojan:
        is_critical = report.spike_ratio >= (spike_threshold * 2.0) or len(report.suspicious_layers) > 1
        badge = get_risk_badge("CRITICAL" if is_critical else "SUSPICIOUS")
        summary_text = (
            f"{badge}\n\n"
            f"• [bold red]Threat Alert:[/bold red] Internal activations exhibited anomalous spikes under perturbation.\n"
            f"• [bold]Peak Activation Layer:[/bold] [cyan]{overall_peak_layer or 'N/A'}[/cyan] ({max(highest_layer_activation, 0.0):.4f})\n"
            f"• [bold]Baseline L_inf:[/bold] {report.baseline_l_inf:.4f}  ──▶  [bold]Peak Fuzzed L_inf:[/bold] {report.peak_fuzzed_l_inf:.4f}\n"
            f"• [bold]Max Spike Multiplier:[/bold] [bold red]{report.spike_ratio:.2f}x[/bold red] (Threshold: {spike_threshold:.1f}x)\n"
            f"• [bold]Flagged Layers ({len(report.suspicious_layers)}):[/bold] {', '.join(report.suspicious_layers) if report.suspicious_layers else 'None'}\n\n"
            f"[dim yellow]Advisory: Disproportionate activation surges under random perturbations indicate dormant Sleeper Agent circuits.[/dim yellow]"
        )
        console.print(
            Panel(
                summary_text,
                title="🚨 Adversarial Trojan Trigger Detected",
                border_style="red" if is_critical else "yellow",
            )
        )
    else:
        badge = get_risk_badge("CLEAN")
        summary_text = (
            f"{badge}\n\n"
            f"• [bold green]Model Integrity Verified:[/bold green] No activation spikes exceeded the {spike_threshold:.1f}x threshold.\n"
            f"• [bold]Peak Monitored Layer:[/bold] [cyan]{overall_peak_layer or 'N/A'}[/cyan]\n"
            f"• [bold]Baseline L_inf:[/bold] {report.baseline_l_inf:.4f}  ──▶  [bold]Peak Fuzzed L_inf:[/bold] {report.peak_fuzzed_l_inf:.4f}\n"
            f"• [bold]Max Spike Multiplier:[/bold] [bold green]{report.spike_ratio:.2f}x[/bold green]\n"
            f"• [bold]Total Fuzzing Iterations:[/bold] {iterations}\n\n"
            f"[dim green]Advisory: Dynamic activations remain bounded within normal Gaussian envelopes.[/dim green]"
        )
        console.print(
            Panel(
                summary_text,
                title="✅ Dynamic Integrity Nominal",
                border_style="green",
            )
        )

    # 12. Optional JSON Telemetry Export
    if json_output:
        iteration_report_payload = []
        for iteration in report.iteration_reports:
            iteration_report_payload.append({
                "iteration": iteration.iteration,
                "input_tag": iteration.input_tag,
                "max_l_inf": iteration.max_l_inf,
                "highest_layer": iteration.highest_layer,
                "layer_stats": {
                    layer_name: {
                        "layer_name": stat.layer_name,
                        "l_inf_norm": stat.l_inf_norm,
                        "mean_activation": stat.mean_activation,
                        "std_activation": stat.std_activation,
                    }
                    for layer_name, stat in iteration.layer_stats.items()
                },
            })

        report_json = {
            "model_path": str(model_path.resolve()),
            "model_name": report.model_name,
            "num_fuzz_samples": report.num_fuzz_samples,
            "baseline_l_inf": report.baseline_l_inf,
            "peak_fuzzed_l_inf": report.peak_fuzzed_l_inf,
            "spike_ratio": report.spike_ratio,
            "spike_threshold": spike_threshold,
            "suspected_trojan": report.suspected_trojan,
            "suspicious_layers": report.suspicious_layers,
            "layer_spike_ratios": report.layer_spike_ratios,
            "seed": seed,
            "peak_layer": overall_peak_layer,
            "layer_peaks": layer_peaks,
            "iteration_reports": iteration_report_payload,
            # Same record format as `aegis scan` (docs/schema/findings-v1.schema.json, finding object).
            "findings": [asdict(f) for f in from_fuzz(str(model_path), report)],
        }
        with open(json_output, "w") as f:
            json.dump(report_json, f, indent=2)
        console.print(f"[dim]Fuzzing report saved to: {json_output}[/dim]")


@app.command(name="disarm")
def disarm_command(
    model_path: Path = typer.Argument(..., help="Path to the .safetensors model to disarm."),
    output_path: Path = typer.Argument(..., help="Where to write the disarmed copy. Must differ from the input."),
    low_bits: int = typer.Option(
        8,
        "--low-bits",
        "-k",
        help="Low mantissa bits to round away. Clamped per dtype to a safe maximum "
             "(float32 16, float16 4, bfloat16 2) unless --allow-unsafe.",
    ),
    strip_metadata: bool = typer.Option(
        False,
        "--strip-metadata",
        help="Also remove the __metadata__ block, where a payload can hide outside the tensors.",
    ),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="Apply --low-bits above the per-dtype safe maximum. Widths that would reach the exponent are always refused.",
    ),
):
    """Round away low-order mantissa bits so hidden payloads are destroyed without detecting them.

    Each weight changes by at most a few units in its last kept place. Use this
    when a scan is inconclusive, or before deploying a model from an untrusted source.
    """
    print_banner()

    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found at: {model_path}")
        raise typer.Exit(code=1)
    if model_path.resolve() == output_path.resolve():
        console.print("[bold red]Error:[/bold red] Output must be a different file from the input.")
        raise typer.Exit(code=1)
    if not TORCH_AVAILABLE:
        console.print(
            Panel(
                "[bold red]PyTorch is required for disarming.[/bold red]\n\n"
                "Install with: [bold green]pip install -e .[fuzzer][/bold green]",
                title="⚠️ PyTorch Engine Missing",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    from aegis.disarm import disarm_file

    try:
        with console.status("[bold green]Rounding low mantissa bits...[/bold green]"):
            report = disarm_file(
                model_path, output_path, low_bits=low_bits,
                strip_metadata=strip_metadata, allow_unsafe=allow_unsafe,
            )
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    applied = ", ".join(f"{dtype} {bits}" for dtype, bits in report.effective_low_bits.items()) or "none"
    console.print(
        Panel(
            f"[bold]Disarmed {report.tensors_disarmed} floating-point tensor(s)[/bold] "
            f"({report.tensors_skipped} non-float tensor(s) copied unchanged).\n\n"
            f"Low bits requested: [cyan]{report.low_bits}[/cyan] · applied per dtype: [cyan]{applied}[/cyan]\n"
            f"Largest per-weight relative change: [cyan]{report.max_relative_change:.3e}[/cyan]\n\n"
            f"Written to: [cyan]{output_path.resolve()}[/cyan]\n"
            "[dim]Re-run `aegis scan` on the output to confirm the structure of the file, "
            "and check accuracy on your own evaluation data.[/dim]",
            title="🛡️ Disarm Complete",
            border_style="green",
        )
    )


@app.command(name="doctor")
def doctor_command():
    """Diagnose local environment for Aegis-Tensor dependencies."""
    print_banner()
    table = Table(title="Aegis-Tensor Environment Diagnostics", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    # Python
    table.add_row("Python", "[green]OK[/green]", sys.version.split()[0])

    # Rust Core
    if CORE_AVAILABLE:
        table.add_row("Rust Core (`aegis_core`)", "[green]OK[/green]", "Native extension loaded")
    else:
        table.add_row("Rust Core (`aegis_core`)", "[yellow]NOT BUILT[/yellow]", "Run `maturin develop` to compile")

    # PyTorch
    if TORCH_AVAILABLE:
        import torch  # type: ignore
        table.add_row("PyTorch Engine", "[green]OK[/green]", f"v{torch.__version__}")
    else:
        table.add_row("PyTorch Engine", "[yellow]MISSING[/yellow]", "Install torch for dynamic Trojan fuzzing")

    console.print(table)


if __name__ == "__main__":
    app()
