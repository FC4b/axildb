//! Binary embedding search via Hamming distance.
//!
//! Packs f32 vectors into binary (1 bit/dim = 32x compression).
//! Two-phase search: fast Hamming scan for candidates, f32 re-rank for final results.

/// Binary vector: 1 bit per dimension, packed into bytes.
///
/// For 384 dims = 48 bytes. For 768 dims = 96 bytes.
#[derive(Debug, Clone)]
pub struct BinaryVector(pub Vec<u8>);

impl BinaryVector {
    /// Binarize an f32 vector: bit = 1 if val > 0, else 0.
    pub fn from_f32(v: &[f32]) -> Self {
        let byte_count = (v.len() + 7) / 8;
        let mut bytes = vec![0u8; byte_count];
        for (i, &val) in v.iter().enumerate() {
            if val > 0.0 {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        BinaryVector(bytes)
    }

    /// Hamming distance: count of differing bits.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /// Hamming similarity: 1.0 - (distance / total_bits).
    pub fn hamming_similarity(&self, other: &Self) -> f32 {
        let total_bits = (self.0.len() * 8) as f32;
        if total_bits == 0.0 {
            return 0.0;
        }
        1.0 - (self.hamming_distance(other) as f32 / total_bits)
    }

    /// Byte size of this binary vector.
    pub fn byte_size(&self) -> usize {
        self.0.len()
    }
}

/// Two-phase binary search: Hamming scan for candidates, f32 cosine re-rank.
///
/// Scan all binary vectors (extremely fast, ~32x less memory than f32)
/// Re-rank top 8*k candidates with f32 cosine similarity
pub fn binary_two_phase_search(
    query_binary: &BinaryVector,
    binary_vectors: &[(usize, BinaryVector)],
    query_f32: &[f32],
    full_vectors: &[(usize, &[f32])],
    top_k: usize,
) -> Vec<(usize, f32)> {
    // Fast Hamming scan
    let candidate_k = top_k.saturating_mul(8).min(binary_vectors.len());
    let mut candidates: Vec<(usize, f32)> = binary_vectors
        .iter()
        .map(|(idx, bv)| (*idx, query_binary.hamming_similarity(bv)))
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(candidate_k);

    // Re-rank with f32 cosine
    let candidate_set: std::collections::HashSet<usize> =
        candidates.iter().map(|(idx, _)| *idx).collect();

    let mut reranked: Vec<(usize, f32)> = full_vectors
        .iter()
        .filter(|(idx, _)| candidate_set.contains(idx))
        .map(|(idx, fv)| (*idx, cosine_sim_f32(query_f32, fv)))
        .collect();

    reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    reranked.truncate(top_k);
    reranked
}

fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    axil_core::util::cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binarize_positive_negative() {
        let v = vec![0.5, -0.3, 0.1, -0.8, 0.0, 0.9, -0.1, 0.2];
        let bv = BinaryVector::from_f32(&v);
        // Expected bits: 1 0 1 0 0 1 0 1 = 0b10100101 = 0xA5
        assert_eq!(bv.0[0], 0xA5);
    }

    #[test]
    fn hamming_identical_is_zero() {
        let v = vec![0.5, -0.3, 0.1, -0.8];
        let bv = BinaryVector::from_f32(&v);
        assert_eq!(bv.hamming_distance(&bv), 0);
        assert!((bv.hamming_similarity(&bv) - 1.0).abs() < 0.01);
    }

    #[test]
    fn hamming_opposite_is_max() {
        let v1 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let v2 = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let bv1 = BinaryVector::from_f32(&v1);
        let bv2 = BinaryVector::from_f32(&v2);
        assert_eq!(bv1.hamming_distance(&bv2), 8);
        assert!((bv1.hamming_similarity(&bv2) - 0.0).abs() < 0.01);
    }

    #[test]
    fn compression_ratio_384_dims() {
        let v = vec![0.0f32; 384];
        let bv = BinaryVector::from_f32(&v);
        // 384 dims: f32 = 1536 bytes, binary = 48 bytes = 32x compression
        assert_eq!(bv.byte_size(), 48);
        assert_eq!(384 * 4 / bv.byte_size(), 32);
    }

    /// Minimal xorshift64* PRNG — keeps the oracle seeded without a `rand` dep.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            bits * 2.0 - 1.0
        }
    }

    fn brute_force_topk(corpus: &[Vec<f32>], query: &[f32], top_k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_sim_f32(query, v)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    fn recall_overlap(approx: &[usize], exact: &[usize]) -> f32 {
        if exact.is_empty() {
            return 1.0;
        }
        let approx_set: std::collections::HashSet<usize> = approx.iter().copied().collect();
        let hits = exact.iter().filter(|i| approx_set.contains(i)).count();
        hits as f32 / exact.len() as f32
    }

    // Binary embedding is the lossiest variant (1 bit/dim, 32x compression),
    // so its recall floor is the lowest of the three. `binary_two_phase_search`
    // is a standalone helper — NOT wired into `VectorEngine::search`.
    #[test]
    fn binary_two_phase_recall_matches_brute_force() {
        let n = 800;
        let dims = 64;
        let top_k = 10;
        let queries = 40;

        let mut rng = Rng::new(0x9A1);
        let corpus: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dims).map(|_| rng.next_f32()).collect())
            .collect();

        let binary: Vec<(usize, BinaryVector)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, BinaryVector::from_f32(v)))
            .collect();
        let full: Vec<(usize, &[f32])> =
            corpus.iter().enumerate().map(|(i, v)| (i, v.as_slice())).collect();

        let mut query_rng = Rng::new(0x5C5);
        let mut total = 0.0f32;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dims).map(|_| query_rng.next_f32()).collect();
            let qb = BinaryVector::from_f32(&q);
            let approx: Vec<usize> = binary_two_phase_search(&qb, &binary, &q, &full, top_k)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            let exact = brute_force_topk(&corpus, &q, top_k);
            total += recall_overlap(&approx, &exact);
        }
        let mean = total / queries as f32;
        // Floor tuned below measured; binary is the lossiest of the variants.
        const BINARY_RECALL_FLOOR: f32 = 0.65;
        assert!(
            mean >= BINARY_RECALL_FLOOR,
            "binary two-phase mean recall@{top_k} {mean:.4} below floor {BINARY_RECALL_FLOOR}"
        );
    }

    #[test]
    fn similar_vectors_have_high_hamming_sim() {
        let a = vec![0.5, 0.3, -0.1, 0.8, 0.2, -0.5, 0.1, 0.9];
        let b = vec![0.4, 0.35, -0.05, 0.75, 0.15, -0.6, 0.05, 0.85];
        let c = vec![-0.5, -0.3, 0.1, -0.8, -0.2, 0.5, -0.1, -0.9]; // opposite

        let ba = BinaryVector::from_f32(&a);
        let bb = BinaryVector::from_f32(&b);
        let bc = BinaryVector::from_f32(&c);

        assert!(ba.hamming_similarity(&bb) > ba.hamming_similarity(&bc));
    }
}
