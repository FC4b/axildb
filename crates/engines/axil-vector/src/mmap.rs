//! Flat binary vector storage format (8b.13).
//!
//! Provides a compact binary file format for vector data that can be
//! memory-mapped by the OS. Current implementation reads via `fs::read()`
//! as a portable fallback; a future version will use `memmap2` for true
//! zero-copy access when the `mmap` feature flag is enabled.
//!
//! File format:
//! - Header: magic (4 bytes) + version (4 bytes) + dims (4 bytes) + count (4 bytes)
//! - Body: count * dims * 4 bytes of contiguous f32 vectors
//! - Index: count * (id_len + id_bytes) for record ID mapping

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use axil_core::RecordId;

/// Magic bytes for the mmap vector file format.
const MAGIC: &[u8; 4] = b"AXVM";
/// Format version.
const VERSION: u32 = 1;
/// Header size: magic(4) + version(4) + dims(4) + count(4)
const HEADER_SIZE: usize = 16;

/// Memory-mapped vector store configuration.
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Enable memory-mapped storage.
    pub enabled: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Write vectors to a flat binary file suitable for memory mapping.
///
/// Returns the path of the written file.
pub fn write_mmap_file(
    path: &Path,
    dims: usize,
    vectors: &[(RecordId, Vec<f32>)],
) -> io::Result<PathBuf> {
    let mmap_path = mmap_path(path);
    let mut file = std::fs::File::create(&mmap_path)?;

    // Header
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(dims as u32).to_le_bytes())?;
    file.write_all(&(vectors.len() as u32).to_le_bytes())?;

    // Vector data (contiguous f32 array)
    for (_id, vec) in vectors {
        for &val in vec {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    // Record ID index
    for (id, _vec) in vectors {
        let id_str = id.to_string();
        let id_bytes = id_str.as_bytes();
        file.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
        file.write_all(id_bytes)?;
    }

    file.flush()?;
    Ok(mmap_path)
}

/// Read vector data from a flat binary file (non-mmap, for portability).
///
/// Returns (dimensions, record_id_to_index_map, flat_vector_data).
pub fn read_mmap_file(path: &Path) -> io::Result<MmapData> {
    let mmap_path = mmap_path(path);
    let data = std::fs::read(&mmap_path)?;

    if data.len() < HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too small for header",
        ));
    }

    // Parse header
    if &data[0..4] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid magic bytes",
        ));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version: {version}"),
        ));
    }
    let dims = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

    // Parse vector data
    let vector_data_size = count * dims * 4;
    if data.len() < HEADER_SIZE + vector_data_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file truncated (vector data)",
        ));
    }

    let vector_bytes = &data[HEADER_SIZE..HEADER_SIZE + vector_data_size];

    // Parse record ID index
    let mut id_map = HashMap::with_capacity(count);
    let mut offset = HEADER_SIZE + vector_data_size;
    for idx in 0..count {
        if offset + 4 > data.len() {
            break;
        }
        let id_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + id_len > data.len() {
            break;
        }
        if let Ok(id_str) = std::str::from_utf8(&data[offset..offset + id_len]) {
            if let Ok(rid) = RecordId::from_string(id_str) {
                id_map.insert(rid, idx);
            }
        }
        offset += id_len;
    }

    Ok(MmapData {
        dims,
        count,
        vector_bytes: vector_bytes.to_vec(),
        id_index: id_map,
    })
}

/// Parsed mmap file data.
pub struct MmapData {
    pub dims: usize,
    pub count: usize,
    /// Flat f32 vector bytes (count * dims * 4).
    pub vector_bytes: Vec<u8>,
    /// Record ID → vector index mapping.
    pub id_index: HashMap<RecordId, usize>,
}

impl MmapData {
    /// Get a vector by its index as an f32 slice.
    pub fn get_vector(&self, idx: usize) -> Option<Vec<f32>> {
        let start = idx * self.dims * 4;
        let end = start + self.dims * 4;
        if end > self.vector_bytes.len() {
            return None;
        }
        let vec: Vec<f32> = self.vector_bytes[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some(vec)
    }

    /// Get a vector by record ID.
    pub fn get_by_id(&self, id: &RecordId) -> Option<Vec<f32>> {
        self.id_index.get(id).and_then(|&idx| self.get_vector(idx))
    }
}

/// Derive the mmap file path from the main database path.
fn mmap_path(main_path: &Path) -> PathBuf {
    let mut p = main_path.as_os_str().to_owned();
    p.push(".mmap");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");

        let id1 = RecordId::new();
        let id2 = RecordId::new();
        let vectors = vec![
            (id1.clone(), vec![0.1f32, 0.2, 0.3]),
            (id2.clone(), vec![0.4, 0.5, 0.6]),
        ];

        write_mmap_file(&path, 3, &vectors).unwrap();
        let data = read_mmap_file(&path).unwrap();

        assert_eq!(data.dims, 3);
        assert_eq!(data.count, 2);
        assert_eq!(data.id_index.len(), 2);

        let v1 = data.get_by_id(&id1).unwrap();
        assert!((v1[0] - 0.1).abs() < 0.001);
        assert!((v1[1] - 0.2).abs() < 0.001);

        let v2 = data.get_by_id(&id2).unwrap();
        assert!((v2[0] - 0.4).abs() < 0.001);
    }

    #[test]
    fn empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.axil");

        write_mmap_file(&path, 3, &[]).unwrap();
        let data = read_mmap_file(&path).unwrap();

        assert_eq!(data.count, 0);
        assert!(data.id_index.is_empty());
    }
}
