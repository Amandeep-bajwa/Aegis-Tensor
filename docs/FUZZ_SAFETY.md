# Fuzzing Untrusted Models Safely

Dynamic fuzzing has to run the model. How much of the model file gets executed
depends on how the model is loaded. Prefer the paths at the top of this table.

| How you run `aegis fuzz` | Code from the file that executes | Isolation |
|---|---|---|
| `.safetensors` + `--arch package.module:Class` | None. The file holds only tensors. | Not needed |
| `.safetensors` + `--hf-config <repo or dir>` | None. `config.json` is read with `trust_remote_code=False`. | Not needed |
| PyTorch state dict (`.pt`) + `--arch`/`--hf-config` | None. Loaded with `torch.load(weights_only=True)`. | Not needed |
| Pickled `nn.Module` + `--trust-pickle` | **All of it.** Unpickling runs the file's code. | Child process in a sandbox, see below |
| Pickled `nn.Module` without `--trust-pickle` | None. The load is refused. | — |
| TorchScript | None. Refused: PyTorch disallows forward hooks on TorchScript. | — |

`--arch` imports *your* code, so the import path must point at code you trust. The
architecture is the trusted half; only the weights come from the untrusted file.

## What the `--trust-pickle` sandbox does

The pickle is loaded and fuzzed in a separate process (`aegis.sandbox_worker`), and only
a JSON report comes back.

**macOS** (`sandbox-exec`):
- All network access is denied.
- Writes are limited to a throwaway directory that is deleted afterwards.
- Only the Python installation may be executed, so `os.system` and `subprocess` cannot start `/bin/sh` or other programs.

**Linux** (`unshare -rn`):
- The process gets a new, empty network namespace, so it has no network.
- Filesystem and exec are **not** restricted.

**Both:**
- The child gets a minimal environment, so tokens and cloud credentials in your shell (`HF_TOKEN`, `AWS_*`) are not passed through.
- `HOME` and `TMPDIR` point at the throwaway directory.
- Python-level socket calls are disabled as a second layer.
- The process is killed after `--sandbox-timeout` seconds (default 1800).

If neither mechanism is available, `--trust-pickle` refuses to run. It never falls back
to an unsandboxed load. On some Linux hosts unprivileged user namespaces are disabled
(for example Ubuntu 24.04 with AppArmor restrictions), and `--trust-pickle` is refused there.

## Residual risk

The sandbox limits what a malicious pickle can do to your machine. It does not make the
pickle safe. With `--trust-pickle`, a malicious file can still:

1. **Read any file your user can read.** Without network it cannot send the data anywhere during the run. On Linux it can still write the data to disk.
2. **Write files on Linux** anywhere your user can write. This includes planting code that runs later, such as shell profiles or Python site-packages. On macOS writes are confined to the throwaway directory.
3. **Spawn programs on Linux.** Only network access is removed there.
4. **Use CPU and memory** until the timeout.
5. **Forge its own fuzz report.** The payload runs inside the process that measures it, so it can tamper with the hooks or the result. Treat a clean report from `--trust-pickle` as weak evidence. A suspicious report is still meaningful.
6. **Exploit the sandbox itself.** `sandbox-exec` and user namespaces are defence in depth, not a security boundary built for hostile code. Kernel or sandbox escapes are out of scope.

For a file you do not trust, use `aegis scan` first. If you need dynamic analysis,
convert the weights to `.safetensors` on a disposable machine and fuzz them with `--arch`
or `--hf-config`.
