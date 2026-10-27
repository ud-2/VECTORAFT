use std::cmp::Ordering;
use std::collections::BinaryHeap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Calculates the Euclidean length (L2 norm) of a vector.
pub fn l2_norm(vector: &[f32]) -> f32 {
    let sum_of_squares: f32 = vector.iter().map(|&x| x * x).sum();
    sum_of_squares.sqrt()
}

/// Normalizes a vector in-place to make its length equal to 1.0.
pub fn normalize(vector: &mut [f32]) {
    let norm = l2_norm(vector);
    if norm > 0.0 {
        for element in vector.iter_mut() {
            *element /= norm;
        }
    }
}

/// Computes the dot product, automatically picking hardware-accelerated SIMD if supported.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dimension mismatch: vectors must have the same length."
    );

    #[cfg(target_arch = "x86_64")]
    {
        // CPU feature detection at runtime
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    // Fallback: standard scalar iterator
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum()
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the dot product between two slices using AVX2 and FMA instructions.
///
/// # Safety
/// This function executes raw CPU intrinsics. The CPU target must support AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum_register = _mm256_setzero_ps();

    let chunks = len / 8;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        // Pointer arithmetic and raw memory loads require an unsafe block.
        let (va, vb) = unsafe {
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            (va, vb)
        };

        // Arithmetic operations on registers are safe.
        sum_register = _mm256_fmadd_ps(va, vb, sum_register);
    }

    // Extract the 8 partial sums to RAM.
    let mut buffer = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buffer.as_mut_ptr(), sum_register);
    }

    let mut total: f32 = buffer.iter().sum();

    // Remainder loop for slices not divisible by 8.
    let remainder = len % 8;
    let remainder_offset = chunks * 8;
    for i in 0..remainder {
        total += a[remainder_offset + i] * b[remainder_offset + i];
    }

    total
}

/// Represents an item returned by a search query.
#[derive(Debug, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Internal wrapper to compare search results inside a Min-Heap.
#[derive(PartialEq)]
struct MinHeapElement {
    id: u64,
    score: f32,
}

// Implement Eq because we guarantee our scores can be compared.
impl Eq for MinHeapElement {}

// We reverse the comparison to turn Rust's default Max-Heap into a Min-Heap!
impl Ord for MinHeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Inverse order: other compared to self!
        other.score.total_cmp(&self.score)
    }
}

impl PartialOrd for MinHeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Represents a single vector entry stored in the database.
#[derive(Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: u64,
    pub values: Vec<f32>,
}

/// The core in-memory vector storage engine.
#[derive(Serialize, Deserialize)]
pub struct VectorStore {
    dimension: usize,
    entries: Vec<VectorEntry>,
}

impl VectorStore {
    /// Creates a new empty VectorStore with a fixed dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: Vec::new(),
        }
    }

    /// Inserts a vector into the store. It is automatically normalized.
    pub fn insert(&mut self, id: u64, mut values: Vec<f32>) {
        assert_eq!(
            values.len(),
            self.dimension,
            "Inserted vector dimension does not match store dimension."
        );

        normalize(&mut values);
        self.entries.push(VectorEntry { id, values });
    }

    /// Searches for the top_k most similar vectors to the query.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query vector dimension does not match store dimension."
        );

        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }

        // Normalize the query vector to ensure dot product equals cosine similarity.
        let mut normalized_query = query.to_vec();
        normalize(&mut normalized_query);

        // We use a Min-Heap of capacity (top_k + 1) to keep only the best candidates.
        let mut heap: BinaryHeap<MinHeapElement> = BinaryHeap::with_capacity(top_k + 1);

        for entry in &self.entries {
            let score = dot_product(&normalized_query, &entry.values);

            heap.push(MinHeapElement {
                id: entry.id,
                score,
            });

            // If we exceed top_k elements, drop the one with the lowest score!
            if heap.len() > top_k {
                heap.pop();
            }
        }

        // Extract results from heap and sort them from highest to lowest score.
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|item| SearchResult {
                id: item.id,
                score: item.score,
            })
            .collect();

        // Sort descending: highest score first.
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results
    }

    /// Serializes and saves the current vector store to a binary file on disk.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Loads and reconstructs a vector store from a binary snapshot file on disk.
    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let store: Self = bincode::deserialize_from(reader)?;
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_identical_vectors() {
        let mut v1 = vec![3.0, 4.0];
        let mut v2 = vec![3.0, 4.0];

        normalize(&mut v1);
        normalize(&mut v2);

        let similarity = dot_product(&v1, &v2);

        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal_vectors() {
        let mut v1 = vec![1.0, 0.0];
        let mut v2 = vec![0.0, 1.0];

        normalize(&mut v1);
        normalize(&mut v2);

        let similarity = dot_product(&v1, &v2);

        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_store_top_k_search() {
        let mut store = VectorStore::new(3);

        // Document 1: Points towards the positive X axis
        store.insert(1, vec![10.0, 0.0, 0.0]);
        // Document 2: Points roughly between X and Y
        store.insert(2, vec![5.0, 5.0, 0.0]);
        // Document 3: Points strictly towards the Z axis (completely different direction)
        store.insert(3, vec![0.0, 0.0, 8.0]);

        // Query: A vector pointing towards X (should match Document 1 best, then Document 2)
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2);

        assert_eq!(results.len(), 2);

        // Best match must be Document 1 with a cosine similarity of ~1.0
        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 1.0).abs() < 1e-5);

        // Second best match must be Document 2
        assert_eq!(results[1].id, 2);
    }

    #[test]
    fn test_avx2_matches_scalar_dot_product() {
        let dimension = 1536;

        // Generate synthetic vector data: [0.0, 1.0, 2.0, ... 1535.0]
        let mut v1: Vec<f32> = (0..dimension).map(|i| (i as f32) * 0.001).collect();
        let mut v2: Vec<f32> = (0..dimension).map(|i| (i as f32) * 0.002).collect();

        normalize(&mut v1);
        normalize(&mut v2);

        // Compute using standard scalar method directly
        let scalar_result: f32 = v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum();

        // Compute using our automated SIMD-enabled dot_product
        let simd_result = dot_product(&v1, &v2);

        // The results must match within floating-point tolerance
        let difference = (scalar_result - simd_result).abs();
        assert!(
            difference < 1e-5,
            "SIMD and scalar calculations diverged! Diff: {}",
            difference
        );
    }

    #[test]
    fn test_vector_store_persistence_roundtrip() {
        let snapshot_file = "test_snapshot.bin";
        let mut original_store = VectorStore::new(3);

        original_store.insert(42, vec![1.0, 0.0, 0.0]);
        original_store.insert(99, vec![0.0, 1.0, 0.0]);

        // Save to disk
        original_store.save_to_disk(snapshot_file).expect("Failed to save snapshot");

        // Load into a completely new instance
        let loaded_store = VectorStore::load_from_disk(snapshot_file).expect("Failed to load snapshot");

        // Clean up temporary test file
        let _ = std::fs::remove_file(snapshot_file);

        // Verify loaded store has same dimension and finds the vectors
        assert_eq!(loaded_store.dimension, 3);
        let results = loaded_store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }
}