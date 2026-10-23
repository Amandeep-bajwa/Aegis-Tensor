"""Severity levels and the exit-code policy for `aegis scan`.

Exit codes:
    0  no finding at or above the --fail-on level (findings below it are still reported)
    1  at least one finding at or above the --fail-on level
    2  the scan could not run (missing file, unreadable or malformed container, missing core)
"""

from typing import Iterable, Optional

LEVELS = ("low", "medium", "high", "critical")
RANK = {level: index for index, level in enumerate(LEVELS)}
DEFAULT_FAIL_ON = "medium"

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

# Tensor rule codes come from TensorScanResult.rules.
TENSOR_RULE_SEVERITY = {
    "unreadable": "high",
    "exponent_lane": "high",
    "out_of_range": "high",
    "low_plane": "high",
    "pooled_entropy": "medium",
    "benford": "low",
}
# Header channels come from FormatFinding.channel.
HEADER_CHANNEL_SEVERITY = {
    "metadata": "high",
    "header_whitespace": "medium",
    "tensor_name": "medium",
}
# Pickle severities come from PickleFinding.severity.
PICKLE_SEVERITY = {
    "CRITICAL": "critical",
    "SUSPICIOUS": "medium",
}


def worst(levels: Iterable[Optional[str]]) -> Optional[str]:
    """Highest level in ``levels``, or None when there are none."""
    present = [level for level in levels if level is not None]
    return max(present, key=RANK.__getitem__, default=None)


def tensor_severity(result) -> Optional[str]:
    """Highest severity among a tensor's triggered rules. Unknown codes count as medium."""
    return worst(TENSOR_RULE_SEVERITY.get(rule, "medium") for rule in result.rules)


def exit_code_for(level: Optional[str], fail_on: str) -> int:
    """EXIT_FINDINGS when ``level`` is at or above ``fail_on``, otherwise EXIT_CLEAN."""
    if level is not None and RANK[level] >= RANK[fail_on]:
        return EXIT_FINDINGS
    return EXIT_CLEAN
