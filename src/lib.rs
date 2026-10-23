use half::{bf16, f16};
use memmap2::MmapOptions;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::fs::File;

/// Result for an individual scanned tensor in a .safetensors file.
#[pyclass]
#[derive(Clone, Debug, serde::Serialize)]
pub struct TensorScanResult {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub num_elements: usize,
    #[pyo3(get)]
    pub entropy: f64,
    #[pyo3(get)]
    pub benford_mad: f64,
    /// Bias-corrected entropy of each byte lane (index 0 = least significant byte).
    /// Empty for non-float tensors.
    #[pyo3(get)]
    pub lane_entropies: Vec<f64>,
    /// Entropy of the sign/exponent byte lane. Near 8 bits means the float values
    /// were overwritten with random-looking data.
    #[pyo3(get)]
    pub exponent_lane_entropy: Option<f64>,
    /// Lowest entropy across the two least significant mantissa bit planes. Clean
    /// trained weights are close to 8 bits; structured payloads pull this down.
    #[pyo3(get)]
    pub low_plane_entropy: Option<f64>,
    /// Fraction of float elements that are non-finite or implausibly large.
    /// Near zero for trained weights; large for overwritten or random data.
    #[pyo3(get)]
    pub out_of_range_fraction: Option<f64>,
    #[pyo3(get)]
    pub is_suspicious: bool,
    #[pyo3(get)]
    pub anomaly_reasons: Vec<String>,
    /// Stable codes, one per entry in `anomaly_reasons`: pooled_entropy, exponent_lane,
    /// out_of_range, low_plane, benford, unreadable. Callers can rely on these; the reason
    /// text is for people and may change wording.
    #[pyo3(get)]
    pub rules: Vec<String>,
}

#[pymethods]
impl TensorScanResult {
    fn __repr__(&self) -> String {
        format!(
            "<TensorScanResult name='{}' shape={:?} entropy={:.4} benford_mad={:.4} suspicious={}>",
            self.name, self.shape, self.entropy, self.benford_mad, self.is_suspicious
        )
    }
}

/// Calculate Shannon entropy over raw byte slices.
///
/// Returns a value between 0.0 (uniform single byte) and 8.0 (pure random/encrypted).
#[pyfunction]
pub fn shannon_entropy_bytes(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let total = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Return the highest Shannon entropy found in any chunk of the input.
///
/// The final chunk may be shorter than `chunk_size`. Empty input and a zero
/// chunk size return 0.0 rather than attempting an invalid partition.
#[pyfunction]
pub fn shannon_entropy_chunked(data: &[u8], chunk_size: usize) -> f64 {
    if data.is_empty() || chunk_size == 0 {
        return 0.0;
    }

    data.chunks(chunk_size)
        .map(shannon_entropy_bytes)
        .fold(0.0, f64::max)
}

fn mantissa_bit_plane_entropy_for(data: &[u8], bit: u32) -> f64 {
    let float_count = data.len() / 4;
    if float_count == 0 {
        return 0.0;
    }

    let mut packed_bits = vec![0u8; float_count.div_ceil(8)];
    for (index, chunk) in data.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mantissa_bit = ((word & 0x007F_FFFF) >> bit) & 1;
        packed_bits[index / 8] |= (mantissa_bit as u8) << (index % 8);
    }

    shannon_entropy_bytes(&packed_bits)
}

/// Compute entropy for the two least-significant float32 mantissa bit planes.
///
/// Input is interpreted as little-endian float32 words. The returned tuple is
/// `(bit_0_entropy, bit_1_entropy)`, with each plane packed into bytes before
/// calculating Shannon entropy. Trailing bytes that do not form a full word
/// are ignored.
#[pyfunction]
pub fn mantissa_bit_plane_entropy(data: &[u8]) -> (f64, f64) {
    (
        mantissa_bit_plane_entropy_for(data, 0),
        mantissa_bit_plane_entropy_for(data, 1),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenfordStats {
    pub mad: f64,
    pub chi_square: f64,
    pub valid_count: usize,
}

/// Compute Benford's Law Mean Absolute Deviation (MAD) for leading digits of float32 values.
///
/// Benford's distribution for first non-zero digit d in {1..9}:
/// P(d) = log10(1 + 1/d)
#[pyfunction]
pub fn benford_law_mad(floats: Vec<f32>) -> f64 {
    calculate_benford_mad_slice(&floats)
}

/// Compute Benford's Law MAD and chi-square statistic together for a float slice.
#[pyfunction]
pub fn benford_law_stats(floats: Vec<f32>) -> (f64, f64) {
    let stats = calculate_benford_stats(&floats);
    (stats.mad, stats.chi_square)
}

fn calculate_benford_mad_slice(floats: &[f32]) -> f64 {
    calculate_benford_stats(floats).mad
}

fn calculate_benford_stats(floats: &[f32]) -> BenfordStats {
    let mut digit_counts = [0usize; 10]; // index 1..=9
    let mut valid_count = 0usize;

    for &val in floats {
        if !val.is_finite() || val == 0.0 {
            continue;
        }

        let abs_val = val.abs();
        let leading_digit = extract_leading_digit(abs_val);
        if (1..=9).contains(&leading_digit) {
            digit_counts[leading_digit] += 1;
            valid_count += 1;
        }
    }

    if valid_count < 100 {
        // Insufficient sample size to reliably apply Benford's Law
        return BenfordStats {
            mad: 0.0,
            chi_square: 0.0,
            valid_count,
        };
    }

    let mut sum_absolute_deviations = 0.0;
    let mut chi_square = 0.0;

    for d in 1..=9 {
        let expected_p = (1.0 + 1.0 / (d as f64)).log10();
        let observed_p = digit_counts[d] as f64 / valid_count as f64;
        let expected_count = expected_p * valid_count as f64;

        sum_absolute_deviations += (observed_p - expected_p).abs();
        if expected_count > 0.0 {
            chi_square += ((digit_counts[d] as f64 - expected_count).powi(2)) / expected_count;
        }
    }

    BenfordStats {
        mad: sum_absolute_deviations / 9.0,
        chi_square,
        valid_count,
    }
}

fn decode_f32_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_f16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn decode_bf16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

#[inline]
fn extract_leading_digit(mut val: f32) -> usize {
    if val >= 1.0 {
        while val >= 10.0 {
            val /= 10.0;
        }
        val as usize
    } else {
        while val < 1.0 && val > 0.0 {
            val *= 10.0;
        }
        val as usize
    }
}

/// Minimum tensor size (bytes) for the pooled whole-tensor entropy rule.
const MIN_BLOB_BYTES: usize = 4096;
/// Minimum element count for the per-lane rule. Bias correction keeps clean small
/// tensors near 2.7 bits in the sign/exponent lane, so this can stay low.
const MIN_LANE_ELEMENTS: usize = 256;
/// Minimum element count for the low-bit-plane rule. The packed plane must hold
/// at least 4096 bytes, otherwise the entropy estimate is too noisy to use.
const MIN_PLANE_ELEMENTS: usize = 32_768;
/// Magnitude above which a trained weight is implausible. Random bit patterns
/// land above this about 45% of the time, while trained weights almost never do.
const MAX_PLAUSIBLE_MAGNITUDE: f32 = 1.0e4;
/// Fraction of out-of-range elements (non-finite or above the magnitude cap) that
/// marks a tensor as overwritten. Clean tensors measured zero in the sweep.
const OUT_OF_RANGE_FRACTION: f64 = 0.001;
/// Minimum number of valid leading digits before Benford's MAD is reported.
const MIN_BENFORD_SAMPLES: usize = 5_000;

/// Miller-Madow bias-corrected Shannon entropy of a byte stream, in bits.
///
/// Plug-in entropy underestimates small samples, so a short random blob would
/// otherwise look less random than it is. The result is capped at 8 bits.
fn corrected_entropy(symbols: &[u8]) -> f64 {
    let n = symbols.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &s in symbols {
        counts[s as usize] += 1;
    }

    let total = n as f64;
    let mut entropy = 0.0;
    let mut observed = 0usize;
    for &count in &counts {
        if count > 0 {
            observed += 1;
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    let corrected = entropy + (observed as f64 - 1.0) / (2.0 * total * std::f64::consts::LN_2);
    corrected.min(8.0)
}

/// Bytes at position `lane` of every `width`-byte element (little-endian layout).
fn byte_lane(data: &[u8], width: usize, lane: usize) -> Vec<u8> {
    data.chunks_exact(width).map(|chunk| chunk[lane]).collect()
}

/// Bias-corrected entropy of bit `plane` of every element, packed into bytes.
///
/// Works on 16- and 32-bit words. Bits 0 and 1 are mantissa bits in f16, bf16
/// and f32 alike, so the same plane index means the same thing for all three.
fn plane_entropy(data: &[u8], width: usize, plane: u32) -> f64 {
    let count = data.len() / width;
    let mut packed = vec![0u8; count.div_ceil(8)];
    for (index, chunk) in data.chunks_exact(width).enumerate() {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        let bit = ((word >> plane) & 1) as u8;
        packed[index / 8] |= bit << (index % 8);
    }
    corrected_entropy(&packed)
}

/// Fraction of elements whose bit `plane` is set.
///
/// A constant plane carries no payload. Models stored at lower precision and
/// upcast to float32 have all-zero low bits, and so do tensors this tool disarms.
/// The low-plane rule only examines planes that are actually mixed.
fn plane_one_fraction(data: &[u8], width: usize, plane: u32) -> f64 {
    let mut ones = 0usize;
    let mut count = 0usize;
    for chunk in data.chunks_exact(width) {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        ones += ((word >> plane) & 1) as usize;
        count += 1;
    }
    if count == 0 { 0.0 } else { ones as f64 / count as f64 }
}

/// True when most sampled elements hold distinct values. Constant or heavily
/// quantized tensors (e.g. LayerNorm weights initialized to 1.0) have skewed low
/// bits without any payload, so the plane rule must skip them.
fn has_diverse_values(data: &[u8], width: usize) -> bool {
    const SAMPLE: usize = 1 << 20;
    let mut seen = std::collections::HashSet::new();
    let mut sampled = 0usize;
    for chunk in data.chunks_exact(width).take(SAMPLE) {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        seen.insert(word);
        sampled += 1;
    }
    sampled > 0 && seen.len() * 2 >= sampled
}

/// Thresholds shared by every tensor rule.
#[derive(Clone, Copy)]
struct Thresholds {
    entropy: f64,
    benford: Option<f64>,
    exponent: f64,
    plane: f64,
}

/// Apply every tensor rule to one tensor's raw little-endian bytes. Used for safetensors
/// files and, through `scan_tensor_bytes`, for tensors read from other formats.
fn analyze_tensor(name: &str, dtype: Dtype, shape: Vec<usize>, data_bytes: &[u8], t: Thresholds) -> TensorScanResult {
    let num_elements: usize = shape.iter().product();

    let entropy = shannon_entropy_bytes(data_bytes);

    let is_float = matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16);
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    };

    let float_values = match dtype {
        Dtype::F32 => Some(decode_f32_bytes(data_bytes)),
        Dtype::F16 => Some(decode_f16_bytes(data_bytes)),
        Dtype::BF16 => Some(decode_bf16_bytes(data_bytes)),
        _ => None,
    };
    let out_of_range_fraction = float_values
        .as_ref()
        .filter(|values| !values.is_empty())
        .map(|values| {
            let bad = values
                .iter()
                .filter(|v| !v.is_finite() || v.abs() > MAX_PLAUSIBLE_MAGNITUDE)
                .count();
            bad as f64 / values.len() as f64
        });
    let benford_stats = float_values.as_ref().map(|values| calculate_benford_stats(values));
    let benford_mad = benford_stats.map(|s| s.mad).unwrap_or(0.0);

    let lane_entropies: Vec<f64> = if is_float {
        (0..width)
            .map(|lane| corrected_entropy(&byte_lane(data_bytes, width, lane)))
            .collect()
    } else {
        Vec::new()
    };

    let exponent_lane_entropy = if is_float && num_elements >= MIN_LANE_ELEMENTS {
        lane_entropies.last().copied()
    } else {
        None
    };

    // Only mixed planes are examined. A constant plane (upcast low precision,
    // or bits zeroed by `aegis disarm`) is not a payload.
    let low_plane_entropy = if is_float
        && num_elements >= MIN_PLANE_ELEMENTS
        && has_diverse_values(data_bytes, width)
    {
        (0..2u32)
            .filter(|&plane| {
                let ones = plane_one_fraction(data_bytes, width, plane);
                ones > 0.01 && ones < 0.99
            })
            .map(|plane| plane_entropy(data_bytes, width, plane))
            .reduce(f64::min)
    } else {
        None
    };

    let mut anomaly_reasons = Vec::new();
    let mut rules: Vec<String> = Vec::new();

    // Rule 1: whole-tensor entropy (encrypted or compressed blob).
    if data_bytes.len() >= MIN_BLOB_BYTES && entropy > t.entropy {
        rules.push("pooled_entropy".to_string());
        anomaly_reasons.push(format!(
            "Unusually high Shannon entropy ({:.4} > threshold {:.4}) - possible encrypted/compressed payload",
            entropy, t.entropy
        ));
    }

    // Rule 2: sign/exponent lane is random-looking, so the float values were overwritten.
    if let Some(lane_entropy) = exponent_lane_entropy {
        if lane_entropy > t.exponent {
            rules.push("exponent_lane".to_string());
            anomaly_reasons.push(format!(
                "Sign/exponent bytes are near-uniform ({:.4} > threshold {:.4}) - float values overwritten with encrypted or compressed data",
                lane_entropy, t.exponent
            ));
        }
    }

    // Rule 2b: implausible magnitudes or non-finite values (overwritten floats).
    if let Some(fraction) = out_of_range_fraction {
        if num_elements >= MIN_LANE_ELEMENTS && fraction > OUT_OF_RANGE_FRACTION {
            rules.push("out_of_range".to_string());
            anomaly_reasons.push(format!(
                "{:.2}% of values are non-finite or above {:.0} in magnitude - float values overwritten with random data",
                fraction * 100.0,
                MAX_PLAUSIBLE_MAGNITUDE
            ));
        }
    }

    // Rule 3: low mantissa bits are structured (payload hidden in the least significant bits).
    if let Some(plane_entropy) = low_plane_entropy {
        if plane_entropy < t.plane {
            rules.push("low_plane".to_string());
            anomaly_reasons.push(format!(
                "Low mantissa bit planes are structured ({:.4} < floor {:.4}) - possible payload hidden in the least significant bits",
                plane_entropy, t.plane
            ));
        }
    }

    // Optional: Benford's Law, only with enough samples and an explicit threshold.
    if let (Some(threshold), Some(stats)) = (t.benford, benford_stats) {
        if stats.valid_count >= MIN_BENFORD_SAMPLES && benford_mad > threshold {
            rules.push("benford".to_string());
            anomaly_reasons.push(format!(
                "Benford's Law MAD deviation ({:.4} > threshold {:.4}) - non-natural digit distribution",
                benford_mad, threshold
            ));
        }
    }

    TensorScanResult {
        name: name.to_string(),
        dtype: format!("{:?}", dtype),
        shape,
        num_elements,
        entropy,
        benford_mad,
        lane_entropies,
        exponent_lane_entropy,
        low_plane_entropy,
        out_of_range_fraction,
        is_suspicious: !anomaly_reasons.is_empty(),
        rules,
        anomaly_reasons,
    }
}

/// Zero-copy scan of a .safetensors model file using memory mapping.
///
/// Each tensor is checked with three rules:
/// 1. Whole-tensor Shannon entropy above `entropy_threshold` (encrypted or
///    compressed blob, for tensors of at least 4 KiB).
/// 2. Sign/exponent byte lane above `exponent_entropy_threshold` (float values
///    overwritten with random-looking data). Clean weights measure about 2.7 bits,
///    and wide-dynamic-range clean weights reach about 6 bits. The default of 6.5
///    therefore detects overwrites covering roughly half of a tensor's floats.
/// 3. Low mantissa bit planes below `plane_floor` (structured data hidden in the
///    least significant bits). Clean weights measure about 8 bits.
///
/// Benford's Law MAD is reported on every tensor, but it only flags a tensor when
/// `benford_mad_threshold` is given. It is noisy on small tensors and is off by default.
#[pyfunction]
#[pyo3(signature = (
    path,
    entropy_threshold = 7.92,
    benford_mad_threshold = None,
    exponent_entropy_threshold = 6.5,
    plane_floor = 7.5
))]
pub fn scan_safetensors(
    path: &str,
    entropy_threshold: f64,
    benford_mad_threshold: Option<f64>,
    exponent_entropy_threshold: f64,
    plane_floor: f64,
) -> PyResult<Vec<TensorScanResult>> {
    let file = File::open(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open model file '{}': {}", path, e)))?;

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| PyIOError::new_err(format!("Failed to memory map file '{}': {}", path, e)))?
    };

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyIOError::new_err(format!("Failed to parse safetensors file: {}", e)))?;

    let tensor_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    // Parallel analysis using Rayon
    let thresholds = Thresholds {
        entropy: entropy_threshold,
        benford: benford_mad_threshold,
        exponent: exponent_entropy_threshold,
        plane: plane_floor,
    };
    let results: Vec<TensorScanResult> = tensor_names
        .par_iter()
        .map(|name| {
            let tensor_view = match tensors.tensor(name) {
                Ok(view) => view,
                Err(error) => {
                    // An unreadable tensor is never silently dropped: a tampered
                    // tensor is exactly the kind a scanner must not skip.
                    return TensorScanResult {
                        name: name.clone(),
                        dtype: "unknown".to_string(),
                        shape: Vec::new(),
                        num_elements: 0,
                        entropy: 0.0,
                        benford_mad: 0.0,
                        lane_entropies: Vec::new(),
                        exponent_lane_entropy: None,
                        low_plane_entropy: None,
                        out_of_range_fraction: None,
                        is_suspicious: true,
                        rules: vec!["unreadable".to_string()],
                        anomaly_reasons: vec![format!(
                            "Tensor data could not be read ({}) - possible tampering",
                            error
                        )],
                    };
                }
            };
            analyze_tensor(name, tensor_view.dtype(), tensor_view.shape().to_vec(), tensor_view.data(), thresholds)
        })
        .collect();

    Ok(results)
}

/// (dtype, bytes per element) for the dtype names used by safetensors.
fn parse_dtype(name: &str) -> Option<(Dtype, usize)> {
    Some(match name {
        "F64" => (Dtype::F64, 8),
        "F32" => (Dtype::F32, 4),
        "F16" => (Dtype::F16, 2),
        "BF16" => (Dtype::BF16, 2),
        "I64" => (Dtype::I64, 8),
        "I32" => (Dtype::I32, 4),
        "I16" => (Dtype::I16, 2),
        "I8" => (Dtype::I8, 1),
        "U64" => (Dtype::U64, 8),
        "U32" => (Dtype::U32, 4),
        "U16" => (Dtype::U16, 2),
        "U8" => (Dtype::U8, 1),
        "BOOL" => (Dtype::BOOL, 1),
        _ => return None,
    })
}

/// Apply the safetensors tensor rules to one tensor read from another format
/// (GGUF, ONNX, Keras). `data` is the tensor's raw little-endian bytes.
#[pyfunction]
#[pyo3(signature = (
    name,
    dtype,
    shape,
    data,
    entropy_threshold = 7.92,
    benford_mad_threshold = None,
    exponent_entropy_threshold = 6.5,
    plane_floor = 7.5
))]
#[allow(clippy::too_many_arguments)]
pub fn scan_tensor_bytes(
    name: &str,
    dtype: &str,
    shape: Vec<usize>,
    data: &[u8],
    entropy_threshold: f64,
    benford_mad_threshold: Option<f64>,
    exponent_entropy_threshold: f64,
    plane_floor: f64,
) -> PyResult<TensorScanResult> {
    let (parsed, width) = parse_dtype(dtype)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype '{}'", dtype)))?;
    let expected = shape.iter().product::<usize>() * width;
    if expected != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tensor '{}': shape {:?} of {} needs {} bytes, got {}",
            name, shape, dtype, expected, data.len()
        )));
    }
    let thresholds = Thresholds {
        entropy: entropy_threshold,
        benford: benford_mad_threshold,
        exponent: exponent_entropy_threshold,
        plane: plane_floor,
    };
    Ok(analyze_tensor(name, parsed, shape, data, thresholds))
}

/// Aegis-Tensor core native extension module.
#[pymodule]
fn aegis_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TensorScanResult>()?;
    m.add_function(wrap_pyfunction!(scan_tensor_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(mantissa_bit_plane_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(benford_law_mad, m)?)?;
    m.add_function(wrap_pyfunction!(benford_law_stats, m)?)?;
    m.add_function(wrap_pyfunction!(scan_safetensors, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_entropy_for_identical_bytes() {
        let zeros = vec![0u8; 1000];
        assert_eq!(shannon_entropy_bytes(&zeros), 0.0);

        let ones = vec![0xFFu8; 500];
        assert_eq!(shannon_entropy_bytes(&ones), 0.0);
    }

    #[test]
    fn test_high_entropy_for_uniform_distribution() {
        // Construct exact uniform distribution of all 256 byte values
        let mut uniform = Vec::new();
        for _ in 0..100 {
            for b in 0..=255u8 {
                uniform.push(b);
            }
        }
        let entropy = shannon_entropy_bytes(&uniform);
        // Shannon entropy of uniform 256 values is exactly 8.0 bits
        assert!((entropy - 8.0).abs() < 1e-4, "Entropy was {}", entropy);
    }

    #[test]
    fn test_chunked_entropy_detects_localized_payload() {
        let mut data = vec![0u8; 1024];
        data.extend((0..=255u8).cycle().take(256));
        data.extend(vec![0u8; 1024]);

        assert_eq!(shannon_entropy_chunked(&data, 256), 8.0);
        assert_eq!(shannon_entropy_chunked(&data, 0), 0.0);
        assert_eq!(shannon_entropy_chunked(&[], 1024), 0.0);
    }

    #[test]
    fn test_mantissa_bit_plane_entropy_ignores_sign_and_exponent() {
        let values = [1.0f32, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        assert_eq!(mantissa_bit_plane_entropy(&bytes), (0.0, 0.0));
        assert_eq!(mantissa_bit_plane_entropy(&[0u8; 3]), (0.0, 0.0));
    }

    #[test]
    fn test_extract_leading_digit() {
        assert_eq!(extract_leading_digit(1.234), 1);
        assert_eq!(extract_leading_digit(9.99), 9);
        assert_eq!(extract_leading_digit(45.67), 4);
        assert_eq!(extract_leading_digit(0.00345), 3);
        assert_eq!(extract_leading_digit(0.789), 7);
    }

    #[test]
    fn test_benford_stats_include_chi_squared_and_threshold_classification() {
        let samples: Vec<f32> = (1..=5000)
            .map(|idx| ((idx % 9) + 1) as f32 + (idx as f32 * 0.01))
            .collect();

        let (mad, chi_square) = benford_law_stats(samples);
        assert!(mad >= 0.0);
        assert!(chi_square >= 0.0);
        assert!(mad > 0.0 || chi_square > 0.0);
    }

    #[test]
    fn test_benford_half_precision_support() {
        let mut values = Vec::new();
        let total_samples = 10000;

        for d in 1..=9 {
            let p_d = (1.0 + 1.0 / (d as f64)).log10();
            let count = (p_d * total_samples as f64).round() as usize;
            for i in 0..count {
                values.push(d as f32 + (i as f32 / count as f32) * 0.9);
            }
        }

        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();

        let f16_decoded = decode_f16_bytes(&f16_bytes);
        let bf16_decoded = decode_bf16_bytes(&bf16_bytes);

        assert!(calculate_benford_mad_slice(&f16_decoded) < 0.005);
        assert!(calculate_benford_mad_slice(&bf16_decoded) < 0.005);
    }

    #[test]
    fn test_benford_mad_low_for_ideal_distribution() {
        // Generate synthetic numbers distributed according to Benford's Law
        let mut benford_samples = Vec::new();
        let total_samples = 10000;

        for d in 1..=9 {
            let p_d = (1.0 + 1.0 / (d as f64)).log10();
            let count = (p_d * total_samples as f64).round() as usize;
            for i in 0..count {
                benford_samples.push(d as f32 + (i as f32 / count as f32) * 0.9);
            }
        }

        let mad = calculate_benford_mad_slice(&benford_samples);
        // Synthetic Benford distribution should have very low MAD (< 0.005)
        assert!(mad < 0.005, "Expected MAD < 0.005, got {}", mad);
    }

    #[test]
    fn test_benford_mad_high_for_uniform_digits() {
        // Uniform digit distribution (e.g. synthetic/encrypted data)
        let mut uniform_digits = Vec::new();
        for _ in 0..1000 {
            for d in 1..=9 {
                uniform_digits.push(d as f32 + 0.5);
            }
        }

        let mad = calculate_benford_mad_slice(&uniform_digits);
        // Non-Benford distribution exhibits significant MAD (> 0.03)
        assert!(mad > 0.03, "Expected MAD > 0.03, got {}", mad);
    }

    /// Deterministic pseudo-random bytes (xorshift), so tests need no RNG crate.
    fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect()
    }

    /// Small-magnitude float32 values that look like trained weights (uniform in +-0.02).
    fn weight_like_f32(count: usize, seed: u64) -> Vec<u8> {
        pseudo_random_bytes(count * 4, seed)
            .chunks_exact(4)
            .flat_map(|c| {
                let unit = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / u32::MAX as f64;
                ((unit * 2.0 - 1.0) as f32 * 0.02).to_le_bytes()
            })
            .collect()
    }

    #[test]
    fn test_corrected_entropy_stays_within_bounds() {
        let uniform: Vec<u8> = (0..=255u8).cycle().take(256 * 50).collect();
        let h = corrected_entropy(&uniform);
        assert!(h <= 8.0 && h > 7.9, "uniform corrected entropy was {}", h);
        assert_eq!(corrected_entropy(&[]), 0.0);
    }

    #[test]
    fn test_exponent_lane_separates_weights_from_random_bytes() {
        let weights = weight_like_f32(20_000, 7);
        let weight_lane = corrected_entropy(&byte_lane(&weights, 4, 3));
        assert!(weight_lane < 5.0, "weight sign/exponent lane was {}", weight_lane);

        let blob = pseudo_random_bytes(80_000, 9);
        let blob_lane = corrected_entropy(&byte_lane(&blob, 4, 3));
        assert!(blob_lane > 7.5, "random sign/exponent lane was {}", blob_lane);
    }

    #[test]
    fn test_low_plane_is_random_for_clean_weights_and_structured_for_text() {
        let weights = weight_like_f32(100_000, 3);
        let clean_plane = plane_entropy(&weights, 4, 0);
        assert!(clean_plane > 7.9, "clean low plane was {}", clean_plane);

        let mut text_bits = weights.clone();
        let payload = b"connect 10.0.0.5:4444 /bin/sh ".repeat(2000);
        for (index, chunk) in text_bits.chunks_exact_mut(4).enumerate() {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let byte = payload[(index / 8) % payload.len()];
            let bit = ((byte >> (index % 8)) & 1) as u32;
            chunk.copy_from_slice(&((word & !1) | bit).to_le_bytes());
        }
        let structured_plane = plane_entropy(&text_bits, 4, 0);
        assert!(structured_plane < 7.5, "structured low plane was {}", structured_plane);
    }

    #[test]
    fn test_constant_tensor_is_not_diverse() {
        let ones: Vec<u8> = std::iter::repeat(1.0f32.to_le_bytes()).take(40_000).flatten().collect();
        assert!(!has_diverse_values(&ones, 4));
        let weights = weight_like_f32(40_000, 5);
        assert!(has_diverse_values(&weights, 4));
    }

    #[test]
    fn test_zeroed_low_bits_are_a_constant_plane_not_a_payload() {
        // Upcast low-precision weights, or bits zeroed by disarming, have constant low planes.
        let masked: Vec<u8> = weight_like_f32(50_000, 11)
            .chunks_exact(4)
            .flat_map(|c| {
                let word = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & !0xFFu32;
                word.to_le_bytes()
            })
            .collect();
        assert_eq!(plane_one_fraction(&masked, 4, 0), 0.0);
        assert_eq!(plane_one_fraction(&masked, 4, 1), 0.0);

        let mixed = weight_like_f32(50_000, 11);
        let fraction = plane_one_fraction(&mixed, 4, 0);
        assert!(fraction > 0.4 && fraction < 0.6, "mixed plane fraction was {}", fraction);
    }
}
