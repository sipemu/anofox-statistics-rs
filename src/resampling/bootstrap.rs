use crate::utils::rng::index;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A stationary bootstrap engine for time series data.
///
/// The stationary bootstrap (Politis & Romano, 1994) generates bootstrap samples
/// by taking random blocks of varying lengths from the original data.
pub struct StationaryBootstrap {
    /// Expected block length (1/q where q is the probability of starting a new block)
    expected_block_length: f64,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl StationaryBootstrap {
    /// Create a new stationary bootstrap engine.
    ///
    /// # Arguments
    /// * `expected_block_length` - Expected length of blocks (typical: n^(1/3) where n is data length)
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(expected_block_length: f64, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            expected_block_length: expected_block_length.max(1.0),
            rng,
        }
    }

    /// Generate a bootstrap sample of the specified length.
    ///
    /// # Arguments
    /// * `data` - The original time series data
    /// * `length` - Length of the bootstrap sample to generate
    pub fn sample(&mut self, data: &[f64], length: usize) -> Vec<f64> {
        self.sample_indices(data.len(), length)
            .into_iter()
            .map(|i| data[i])
            .collect()
    }

    /// Draw the time indices (0-based, into a series of length `n`) of one
    /// bootstrap resample of the given length.
    ///
    /// Applying the same index vector to several series resamples them
    /// jointly, preserving their cross-sectional dependence (as required by
    /// the SPA test and the Model Confidence Set). [`sample`](Self::sample)
    /// is `sample_indices` applied to one series, with the same random stream.
    pub fn sample_indices(&mut self, n: usize, length: usize) -> Vec<usize> {
        if n == 0 || length == 0 {
            return vec![];
        }

        let q = 1.0 / self.expected_block_length;
        let mut result = Vec::with_capacity(length);

        // Start with a random index
        let mut current_idx = index(&mut self.rng, n);

        for _ in 0..length {
            result.push(current_idx);

            // With probability q, start a new block (random jump)
            // Otherwise, continue to next index (wrap around)
            if self.rng.gen::<f64>() < q {
                current_idx = index(&mut self.rng, n);
            } else {
                current_idx = (current_idx + 1) % n;
            }
        }

        result
    }

    /// Generate multiple bootstrap samples.
    pub fn samples(&mut self, data: &[f64], length: usize, n_samples: usize) -> Vec<Vec<f64>> {
        (0..n_samples).map(|_| self.sample(data, length)).collect()
    }
}

/// A circular block bootstrap engine.
///
/// The circular block bootstrap uses fixed-length blocks with wrap-around.
pub struct CircularBlockBootstrap {
    /// Block length
    block_length: usize,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl CircularBlockBootstrap {
    /// Create a new circular block bootstrap engine.
    ///
    /// # Arguments
    /// * `block_length` - Length of each block
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(block_length: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            block_length: block_length.max(1),
            rng,
        }
    }

    /// Generate a bootstrap sample of the specified length.
    pub fn sample(&mut self, data: &[f64], length: usize) -> Vec<f64> {
        self.sample_indices(data.len(), length)
            .into_iter()
            .map(|i| data[i])
            .collect()
    }

    /// Draw the time indices (0-based, into a series of length `n`) of one
    /// bootstrap resample of the given length; see
    /// [`StationaryBootstrap::sample_indices`].
    pub fn sample_indices(&mut self, n: usize, length: usize) -> Vec<usize> {
        if n == 0 || length == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(length);

        while result.len() < length {
            // Random starting point for the block
            let start = index(&mut self.rng, n);

            // Add elements from the block (with wrap-around)
            for offset in 0..self.block_length {
                if result.len() >= length {
                    break;
                }
                result.push((start + offset) % n);
            }
        }

        result
    }

    /// Generate multiple bootstrap samples.
    pub fn samples(&mut self, data: &[f64], length: usize, n_samples: usize) -> Vec<Vec<f64>> {
        (0..n_samples).map(|_| self.sample(data, length)).collect()
    }
}

/// A moving (non-circular) block bootstrap engine.
///
/// Blocks of fixed length `block_length` start uniformly at one of the
/// `n - block_length + 1` positions that keep the block inside the series;
/// `ceil(length / block_length)` blocks are concatenated and truncated to
/// `length` (Kuensch, 1989). This is the scheme of R's `MCS::MCSprocedure`.
/// A block length of 1 (or a block longer than the series) degenerates to
/// the IID bootstrap / a single block.
pub struct MovingBlockBootstrap {
    block_length: usize,
    rng: ChaCha8Rng,
}

impl MovingBlockBootstrap {
    /// Create a new moving block bootstrap engine.
    pub fn new(block_length: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            block_length: block_length.max(1),
            rng,
        }
    }

    /// Generate a bootstrap sample of the specified length.
    pub fn sample(&mut self, data: &[f64], length: usize) -> Vec<f64> {
        self.sample_indices(data.len(), length)
            .into_iter()
            .map(|i| data[i])
            .collect()
    }

    /// Draw the time indices of one bootstrap resample; see
    /// [`StationaryBootstrap::sample_indices`].
    pub fn sample_indices(&mut self, n: usize, length: usize) -> Vec<usize> {
        if n == 0 || length == 0 {
            return vec![];
        }
        let k = self.block_length.min(n);
        let n_starts = n - k + 1;
        let mut result = Vec::with_capacity(length);
        while result.len() < length {
            let start = index(&mut self.rng, n_starts);
            for offset in 0..k {
                if result.len() >= length {
                    break;
                }
                result.push(start + offset);
            }
        }
        result
    }
}

/// Resampling scheme of a (block) bootstrap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BootstrapMethod {
    /// IID (nonparametric) bootstrap: draw single observations with replacement.
    Iid,
    /// Stationary bootstrap (Politis & Romano, 1994): blocks of geometric
    /// length with the given mean, wrapping around the end of the series
    /// (R `boot::tsboot(sim = "geom")`).
    Stationary {
        /// Expected block length (`>= 1`; 1 is the IID bootstrap).
        expected_block_length: f64,
    },
    /// Circular block bootstrap: fixed-length blocks starting anywhere,
    /// wrapping around (R `boot::tsboot(sim = "fixed")`).
    CircularBlock {
        /// Block length (`>= 1`).
        block_length: usize,
    },
    /// Moving (non-circular) block bootstrap, see [`MovingBlockBootstrap`].
    MovingBlock {
        /// Block length (`>= 1`).
        block_length: usize,
    },
}

impl BootstrapMethod {
    /// Validate the block length.
    pub(crate) fn validate(&self) -> crate::error::Result<()> {
        let ok = match *self {
            BootstrapMethod::Iid => true,
            BootstrapMethod::Stationary {
                expected_block_length: l,
            } => l.is_finite() && l >= 1.0,
            BootstrapMethod::CircularBlock { block_length }
            | BootstrapMethod::MovingBlock { block_length } => block_length >= 1,
        };
        if ok {
            Ok(())
        } else {
            Err(crate::error::StatError::InvalidParameter(format!(
                "invalid bootstrap block length in {:?} (must be >= 1)",
                self
            )))
        }
    }
}

/// Draws bootstrap index vectors for any [`BootstrapMethod`].
pub(crate) enum IndexSampler {
    Iid(ChaCha8Rng),
    Stationary(StationaryBootstrap),
    Circular(CircularBlockBootstrap),
    Moving(MovingBlockBootstrap),
}

impl IndexSampler {
    pub(crate) fn new(method: BootstrapMethod, seed: Option<u64>) -> Self {
        match method {
            BootstrapMethod::Iid => IndexSampler::Iid(match seed {
                Some(s) => ChaCha8Rng::seed_from_u64(s),
                None => ChaCha8Rng::from_entropy(),
            }),
            BootstrapMethod::Stationary {
                expected_block_length,
            } => IndexSampler::Stationary(StationaryBootstrap::new(expected_block_length, seed)),
            BootstrapMethod::CircularBlock { block_length } => {
                IndexSampler::Circular(CircularBlockBootstrap::new(block_length, seed))
            }
            BootstrapMethod::MovingBlock { block_length } => {
                IndexSampler::Moving(MovingBlockBootstrap::new(block_length, seed))
            }
        }
    }

    /// Indices of one resample of length `n` from a series of length `n`.
    pub(crate) fn draw(&mut self, n: usize) -> Vec<usize> {
        match self {
            IndexSampler::Iid(rng) => (0..n).map(|_| index(rng, n)).collect(),
            IndexSampler::Stationary(b) => b.sample_indices(n, n),
            IndexSampler::Circular(b) => b.sample_indices(n, n),
            IndexSampler::Moving(b) => b.sample_indices(n, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_bootstrap_length() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut bootstrap = StationaryBootstrap::new(5.0, Some(42));

        let sample = bootstrap.sample(&data, 50);
        assert_eq!(sample.len(), 50);
    }

    #[test]
    fn test_stationary_bootstrap_reproducibility() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        let sample1 = StationaryBootstrap::new(5.0, Some(42)).sample(&data, 50);
        let sample2 = StationaryBootstrap::new(5.0, Some(42)).sample(&data, 50);

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_stationary_bootstrap_empty_data() {
        let data: Vec<f64> = vec![];
        let mut bootstrap = StationaryBootstrap::new(5.0, Some(42));

        let sample = bootstrap.sample(&data, 10);
        assert!(sample.is_empty());
    }

    #[test]
    fn test_circular_block_bootstrap_length() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut bootstrap = CircularBlockBootstrap::new(10, Some(42));

        let sample = bootstrap.sample(&data, 50);
        assert_eq!(sample.len(), 50);
    }

    #[test]
    fn test_circular_block_bootstrap_multiple_samples() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let mut bootstrap = CircularBlockBootstrap::new(5, Some(42));

        let samples = bootstrap.samples(&data, 30, 100);
        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert_eq!(sample.len(), 30);
        }
    }
}
