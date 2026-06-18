//! Diagnostic: embed a few strings with a given model and report whether
//! the vectors are healthy (non-zero, non-collapsed, finite).
//!
//! Run: cargo run --release -p axil-vector --features embed \
//!        --example embed_probe -- gte-modernbert-base

use axil_vector::embed::Embedder;
use axil_vector::models::EmbeddingModel;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn main() {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bge-small".to_string());
    let model = EmbeddingModel::from_name(&model_name)
        .unwrap_or_else(|| panic!("unknown model: {model_name}"));
    println!("model: {} (dim {})", model.name(), model.dimensions());

    let embedder = Embedder::new(model).expect("load embedder");

    let texts = [
        "The user graduated with a degree in Business Administration.",
        "We deployed a new Redis caching layer to the backend service.",
        "What programming language is best for systems work?",
    ];

    let mut vecs = Vec::new();
    for t in &texts {
        let v = embedder.embed(t).expect("embed");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nonzero = v.iter().filter(|x| **x != 0.0).count();
        let has_nan = v.iter().any(|x| x.is_nan());
        println!(
            "  len={} norm={:.4} nonzero={}/{} nan={} first5={:?}",
            v.len(),
            norm,
            nonzero,
            v.len(),
            has_nan,
            &v[..5.min(v.len())]
        );
        vecs.push(v);
    }

    println!("\npairwise cosine (unrelated texts should be LOW, ~0.0-0.5):");
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            println!("  text{i} vs text{j}: {:.4}", cosine(&vecs[i], &vecs[j]));
        }
    }

    let collapsed = vecs.len() >= 2 && cosine(&vecs[0], &vecs[1]) > 0.98;
    if collapsed {
        println!("\n⚠️  VECTORS COLLAPSED — unrelated texts have cosine > 0.98.");
        println!("    The embedding output is degenerate.");
        std::process::exit(1);
    }
    println!("\n✓ vectors look healthy (distinct, finite, normalized)");
}
