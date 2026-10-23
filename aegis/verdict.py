"""One verdict per file, with the next step to take (see SECURITY.md).

  CRITICAL           do not load: a code-execution payload or a file that loads code
  SUSPICIOUS         statistical or structural anomaly: disarm, or get the file elsewhere
  NO KNOWN FINDINGS  nothing known was found; disarm is still recommended, because the
                     rules are public and a payload can be tuned to stay under them

Each finding also says how a confirmed problem should be disclosed: to the platform
first (clearly malicious: never contact the uploader, who may be the attacker), to the
owner privately (probably accidental), or not at all.
"""

from typing import Any, Dict, List, Optional

from aegis.disarm import default_disarm_path

CRITICAL, SUSPICIOUS, NO_KNOWN_FINDINGS = "critical", "suspicious", "no_known_findings"
LABELS = {CRITICAL: "CRITICAL: do not load", SUSPICIOUS: "SUSPICIOUS", NO_KNOWN_FINDINGS: "NO KNOWN FINDINGS"}
DISARMABLE = {"safetensors", "gguf"}
PICKLE_FORMATS = {"pickle", "pytorch-legacy", "zip", "tar", "7z", "npy"}

# How to disclose a confirmed finding, by rule.
PLATFORM, OWNER, NONE = "platform", "owner", "none"
ROUTE = {
    # Code execution, obfuscation and deliberate hiding: the uploader may be the attacker.
    **{rule: PLATFORM for rule in (
        "AEGIS-P001", "AEGIS-P003", "AEGIS-P004", "AEGIS-P006", "AEGIS-P007", "AEGIS-G001", "AEGIS-K001",
        "AEGIS-X001", "AEGIS-T002", "AEGIS-T003", "AEGIS-T004", "AEGIS-H001", "AEGIS-H002", "AEGIS-H003",
        "AEGIS-B001")},
    # Insecure or careless, most likely not malicious.
    **{rule: OWNER for rule in ("AEGIS-H004", "AEGIS-P002", "AEGIS-P008", "AEGIS-K002", "AEGIS-X003")},
}


def route_for(rule: str) -> str:
    return ROUTE.get(rule, NONE)


def file_verdict(path: str, file_format: Optional[str], findings: List[Any], error: Optional[str] = None) -> Dict[str, Any]:
    """Verdict for one scanned file. ``findings`` are aegis.findings.Finding objects."""
    severities = {f.severity for f in findings}
    disarm = f"aegis disarm {path} {default_disarm_path(path)}" if file_format in DISARMABLE else None
    if error:
        return {"level": SUSPICIOUS, "label": LABELS[SUSPICIOUS], "reason": f"the file could not be scanned: {error}",
                "next_step": "Do not load a file Aegis could not read. Get it from another source.", "command": None,
                "disclosure": NONE}
    routes = {route_for(f.rule) for f in findings if f.severity != "low"}
    disclosure = PLATFORM if PLATFORM in routes else OWNER if OWNER in routes else NONE
    if "critical" in severities:
        rules = sorted({f.rule for f in findings if f.severity == "critical"})
        return {"level": CRITICAL, "label": LABELS[CRITICAL], "reason": f"critical finding(s): {', '.join(rules)}",
                "next_step": "Do not load this file. Quarantine it (move it out of every model directory) and report "
                             "it to the platform, not the uploader (SECURITY.md). `aegis report-bundle` prepares "
                             "the report.",
                "command": f"aegis report-bundle {path}", "disclosure": PLATFORM}
    if severities & {"high", "medium"}:
        rules = sorted({f.rule for f in findings if f.severity in ("high", "medium")})
        if disarm:
            step = ("Run `aegis disarm`, then scan the output again. If findings remain (overwritten tensors, "
                    "tampered structure), get the file from another source.")
        else:
            step = "Get the file from another source, or inspect the findings before loading it."
        return {"level": SUSPICIOUS, "label": LABELS[SUSPICIOUS], "reason": f"finding(s): {', '.join(rules)}",
                "next_step": step, "command": disarm, "disclosure": disclosure}
    if disarm:
        step = ("Disarm recommended: no known payload was found, but the rules are public and a payload can be "
                "tuned to stay under them. Disarming removes anything in the low-order bits.")
    elif file_format in PICKLE_FORMATS:
        step = ("No dangerous imports found, but pickle files can run code when loaded. Prefer a safetensors or "
                "GGUF version of this model.")
    else:
        step = "No known findings. This does not prove the file is safe."
    if disclosure == NONE and file_format in PICKLE_FORMATS - {"npy"}:
        disclosure = OWNER  # an insecure format: suggest a safetensors version to the owner
    return {"level": NO_KNOWN_FINDINGS, "label": LABELS[NO_KNOWN_FINDINGS],
            "reason": "no finding at medium severity or above", "next_step": step, "command": disarm,
            "disclosure": disclosure}


def overall(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    order = [NO_KNOWN_FINDINGS, SUSPICIOUS, CRITICAL]
    worst = max((v["level"] for v in verdicts), key=order.index, default=NO_KNOWN_FINDINGS)
    return {"level": worst, "label": LABELS[worst],
            "files": {level: sum(1 for v in verdicts if v["level"] == level) for level in order}}
