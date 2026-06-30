//! Encryption-at-rest overhead benchmark (XChaCha20-Poly1305 on record bodies).
//!
//! Run: `cargo bench -p axil-core --features encryption --bench encryption_benchmarks`
//!
//! Measures (a) the isolated AEAD seal/unseal cost at several body sizes, and
//! (b) the full `Storage` insert/get path with vs without a cipher attached — so
//! the crypto overhead can be read against the redb commit cost it sits next to.
//! Encryption lives at the `Storage` layer (`Storage::with_cipher`), which is the
//! layer these benches drive directly.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use serde_json::json;
use tempfile::TempDir;

use axil_core::crypto::Cipher;
use axil_core::{Record, Storage};

fn bench_cipher() -> Cipher {
    Cipher::from_key_bytes(&[7u8; 32]).expect("32-byte key")
}

/// A record whose serialized JSON body is roughly `target` bytes of payload.
fn record_of_size(target: usize) -> Record {
    Record::new("bench", json!({ "body": "x".repeat(target) }))
}

// (a) Isolated AEAD seal/unseal at several body sizes — the pure crypto cost.
fn bench_aead(c: &mut Criterion) {
    let cipher = bench_cipher();
    let aad = "01ABCDEF01ABCDEF01ABCDEF0123";
    let mut group = c.benchmark_group("aead_seal_unseal");
    for size in [256usize, 1024, 4096, 16384] {
        let plaintext = vec![0xABu8; size];
        let ciphertext = cipher.encrypt(&plaintext, aad).unwrap();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encrypt", size), &plaintext, |b, pt| {
            b.iter(|| black_box(cipher.encrypt(black_box(pt), aad).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("decrypt", size), &ciphertext, |b, ct| {
            b.iter(|| black_box(cipher.decrypt(black_box(ct), aad).unwrap()))
        });
    }
    group.finish();
}

// (b) Full Storage insert path (includes serde + redb write/commit), cleartext
// vs cipher-attached — the overhead in real-operation context.
//
// Each timed iteration inserts the FIRST record into a FRESH store via
// `iter_batched(.., PerIteration)`: the TempDir + `Storage::open` happen in the
// untimed setup, so the measurement is a single stable insert. (Reusing one
// store and inserting across `b.iter` would grow the redb table every iteration,
// folding B-tree growth + index re-serialization into the timing and producing
// noise that can read as "encrypted faster than cleartext".) The setup tuple
// returns the `TempDir` so it outlives — and drops only after — the timed insert.
fn bench_storage_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_insert");
    for &size in &[256usize, 4096] {
        group.bench_with_input(BenchmarkId::new("cleartext", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().unwrap();
                    let storage = Storage::open(dir.path().join("c.axil")).unwrap();
                    (dir, storage, record_of_size(size))
                },
                |(_dir, storage, r)| black_box(storage.insert(black_box(&r)).unwrap()),
                BatchSize::PerIteration,
            );
        });
        group.bench_with_input(BenchmarkId::new("encrypted", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().unwrap();
                    let storage = Storage::open(dir.path().join("e.axil"))
                        .unwrap()
                        .with_cipher(bench_cipher());
                    (dir, storage, record_of_size(size))
                },
                |(_dir, storage, r)| black_box(storage.insert(black_box(&r)).unwrap()),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

// Full Storage get path (includes redb read + serde + decrypt), cleartext vs
// cipher-attached.
fn bench_storage_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_get");
    for &size in &[256usize, 4096] {
        {
            let dir = TempDir::new().unwrap();
            let storage = Storage::open(dir.path().join("c.axil")).unwrap();
            let id = storage.insert(&record_of_size(size)).unwrap();
            group.bench_with_input(BenchmarkId::new("cleartext", size), &size, |b, _| {
                b.iter(|| black_box(storage.get(black_box(&id)).unwrap()))
            });
        }
        {
            let dir = TempDir::new().unwrap();
            let storage = Storage::open(dir.path().join("e.axil"))
                .unwrap()
                .with_cipher(bench_cipher());
            let id = storage.insert(&record_of_size(size)).unwrap();
            group.bench_with_input(BenchmarkId::new("encrypted", size), &size, |b, _| {
                b.iter(|| black_box(storage.get(black_box(&id)).unwrap()))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_aead, bench_storage_insert, bench_storage_get);
criterion_main!(benches);
