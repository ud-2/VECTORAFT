use std::time::Instant;
use vector_engine::{dot_product, VectorStore};

/// Naive scalar dot product implementation used strictly as a baseline comparison.
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn main() {
    println!("=== VectoRaft: SIMD Hardware Acceleration Benchmark ===");

    let dimension: usize = 1536; // Standard OpenAI embedding dimension
    let vector_count: usize = 10_000;

    println!(
        "Generating synthetic dataset: {} vectors of dimension {}...",
        vector_count, dimension
    );

    let mut store = VectorStore::new(dimension);

    // Populate the store with dummy normalized vectors
    for id in 0..vector_count {
        let values: Vec<f32> = (0..dimension)
            .map(|i| ((i + (id as usize)) % 100) as f32 * 0.01)
            .collect();
        store.insert(id as u64, values);
    }

    // Query vector representing a search prompt
    let query: Vec<f32> = (0..dimension).map(|i| (i % 50) as f32 * 0.02).collect();

    println!("Dataset generated. Running performance comparisons...\n");

    // Measure raw scalar dot product (10,000 operations)
    let dummy_a = vec![0.5f32; dimension];
    let dummy_b = vec![0.5f32; dimension];

    let start_scalar = Instant::now();
    let mut scalar_accumulator = 0.0f32;
    for _ in 0..vector_count {
        scalar_accumulator += scalar_dot_product(&dummy_a, &dummy_b);
    }
    let duration_scalar = start_scalar.elapsed();

    // Measure automated SIMD dot product (10,000 operations)
    let start_simd = Instant::now();
    let mut simd_accumulator = 0.0f32;
    for _ in 0..vector_count {
        simd_accumulator += dot_product(&dummy_a, &dummy_b);
    }
    let duration_simd = start_simd.elapsed();

    println!("--- Raw Dot Product (10,000 iterations)");
    println!("Scalar loop execution time : {:?}", duration_scalar);
    println!("SIMD AVX2 execution time   : {:?}", duration_simd);
    let speedup = duration_scalar.as_secs_f64() / duration_simd.as_secs_f64();
    println!("Speedup factor             : {:.2}x faster\n", speedup);

    // Prevent compiler from optimizing away the loops
    assert!((scalar_accumulator - simd_accumulator).abs() < 1e-3);

    // Measure full Top-K search on the VectorStore
    let top_k = 10;
    let start_search = Instant::now();
    let results = store.search(&query, top_k);
    let duration_search = start_search.elapsed();

    println!("--- Full VectorStore Top-K Search");
    println!(
        "Search top {} across {} vectors: {:?}",
        top_k, vector_count, duration_search
    );
    println!("Top 1 result: ID = {}, Score = {:.4}", results[0].id, results[0].score);
}