"""Child-process entry point for sandboxed fuzzing. Run only by aegis.sandbox.

Usage: python -m aegis.sandbox_worker REQUEST.json RESPONSE.json

The OS sandbox is the real control. Blocking sockets here is a second layer that
also covers the case where the sandbox profile is misconfigured.
"""

import json
import sys


def _block_network() -> None:
    import socket

    def refuse(*_args, **_kwargs):
        raise OSError("network access is disabled in the aegis sandbox")

    # Patch methods rather than replacing the class: ssl and others subclass socket.socket.
    for name in ("connect", "connect_ex", "sendto", "bind"):
        setattr(socket.socket, name, refuse)
    socket.getaddrinfo = refuse
    socket.create_connection = refuse


def _probe(request: dict) -> dict:
    """Self-test used by the test suite: report what the sandbox allows."""
    import socket
    import subprocess

    results = {}
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3)
        results["network"] = "allowed"
    except OSError as error:
        results["network"] = f"blocked: {error}"
    try:
        with open(request["outside_path"], "w") as handle:
            handle.write("x")
        results["write_outside"] = "allowed"
    except OSError as error:
        results["write_outside"] = f"blocked: {error}"
    try:
        subprocess.run(["/bin/sh", "-c", "true"], check=True, capture_output=True)
        results["exec_shell"] = "allowed"
    except (OSError, subprocess.CalledProcessError) as error:
        results["exec_shell"] = f"blocked: {error}"
    return {"probe": results}


def main(request_path: str, response_path: str) -> int:
    with open(request_path) as handle:
        request = json.load(handle)
    try:
        if request.get("action") == "probe":
            if request.get("python_block", True):
                _block_network()
            response = _probe(request)
        else:
            import torch  # imported before the block so its own imports are unaffected

            from aegis.modelload import fuzz_model

            _block_network()
            model = torch.load(request["model_path"], map_location="cpu", weights_only=False)
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f"the file holds a {type(model).__name__}, not a torch.nn.Module")
            report = fuzz_model(
                model,
                spike_threshold=request["spike_threshold"],
                iterations=request["iterations"],
                seed=request["seed"],
                input_shape=tuple(request["input_shape"]) if request.get("input_shape") else None,
                vocab_size=request.get("vocab_size"),
                seq_len=request.get("seq_len"),
            )
            response = {"report": report.to_dict()}
    except Exception as error:
        response = {"error": f"{type(error).__name__}: {error}"}
    with open(response_path, "w") as handle:
        json.dump(response, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
