"""Default scan thresholds, shared by the CLI and the calibration benchmark.

They match the defaults of the Rust scan functions (src/lib.rs). Calibration results
record them, so a change here marks those results, and the claims that rest on them, stale.
"""

DEFAULT_THRESHOLDS = {
    "entropy": 7.92,  # pooled whole-tensor entropy, bits per byte
    "exponent": 6.5,  # sign/exponent byte-lane entropy
    "plane": 7.5,  # low mantissa bit-plane entropy floor
    "benford": None,  # Benford MAD; off by default
}
