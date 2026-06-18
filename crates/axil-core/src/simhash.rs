//! Lexical SimHash for near-duplicate collapse at recall time.
//!
//! A 64-bit SimHash over character 4-grams lets recall detect near-identical
//! results — the same text re-stored, or case/whitespace/punctuation/tiny-edit
//! variants of it — without a vector index or an embedding model, so the scarce
//! `top_k` slots aren't spent on redundancy. This complements (does not replace)
//! the on-insert vector supersede, which is per-table and requires vectors;
//! SimHash here is synchronous and dependency-free.
//!
//! The fingerprint is intentionally lexical and the recall threshold deliberately
//! tight: it collapses text that differs only in case, whitespace, or a few
//! characters, but does *not* treat semantically-equivalent paraphrases as
//! duplicates (a one-word synonym swap already moves ~10 bits — well past the
//! default threshold; that is the embedding path's job). Identifiers, error
//! codes, and code fragments stay distinct because a few changed characters move
//! several SimHash bits.

/// Normalize text for fingerprinting: lowercase every token and collapse all
/// whitespace runs to single spaces. Two strings that differ only in case or
/// spacing normalize to the same value (and hence the same fingerprint).
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 64-bit SimHash over character 4-grams using bit-voting.
///
/// Each 4-gram is hashed (FNV-1a) to 64 bits; bit `i` of the output is set when
/// more grams voted 1 than 0 at position `i`. Text shorter than one full gram is
/// hashed whole. Empty text returns 0.
pub fn simhash(text: &str) -> u64 {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return 0;
    }
    let mut votes = [0i32; 64];
    let mut vote = |h: u64| {
        for (i, v) in votes.iter_mut().enumerate() {
            if (h >> i) & 1 == 1 {
                *v += 1;
            } else {
                *v -= 1;
            }
        }
    };
    if chars.len() < 4 {
        let gram: String = chars.iter().collect();
        vote(fnv1a_64(gram.as_bytes()));
    } else {
        for window in chars.windows(4) {
            let gram: String = window.iter().collect();
            vote(fnv1a_64(gram.as_bytes()));
        }
    }
    let mut fp = 0u64;
    for (i, v) in votes.iter().enumerate() {
        if *v > 0 {
            fp |= 1u64 << i;
        }
    }
    fp
}

/// Hamming distance between two fingerprints (number of differing bits).
#[inline]
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_zero_distance() {
        let a = simhash(&normalize("Fixed the auth timeout bug in login flow"));
        let b = simhash(&normalize("Fixed the auth timeout bug in login flow"));
        assert_eq!(hamming(a, b), 0);
    }

    #[test]
    fn case_and_whitespace_normalize_to_same_fingerprint() {
        let a = simhash(&normalize("Fixed   the AUTH timeout bug"));
        let b = simhash(&normalize("fixed the auth timeout bug"));
        assert_eq!(hamming(a, b), 0);
    }

    #[test]
    fn near_duplicate_is_closer_than_unrelated() {
        // A one-word edit is not bit-for-bit identical (SimHash is conservative,
        // which is why the recall default threshold only collapses near-exact
        // restatements) — but it must still land much closer than unrelated text.
        let base = simhash(&normalize(
            "Decided to use RRF fusion for recall because it is rank-based and robust",
        ));
        let near = simhash(&normalize(
            "Decided to use RRF fusion for recall since it is rank-based and robust",
        ));
        let far = simhash(&normalize(
            "The dependency doc memory pins versions from the lockfile closure",
        ));
        assert!(
            hamming(base, near) < hamming(base, far),
            "near-dup ({}) must be closer than unrelated ({})",
            hamming(base, near),
            hamming(base, far),
        );
    }

    #[test]
    fn unrelated_text_is_far_apart() {
        let a = simhash(&normalize(
            "Implemented the SCIP code-graph ingest with prost protobuf parsing",
        ));
        let b = simhash(&normalize(
            "The dependency doc memory pins versions from the lockfile closure",
        ));
        assert!(hamming(a, b) > 12, "unrelated distance was {}", hamming(a, b));
    }

    #[test]
    fn empty_and_short_text_do_not_panic() {
        assert_eq!(simhash(""), 0);
        let _ = simhash("ab");
        let _ = simhash(&normalize("  "));
    }
}
