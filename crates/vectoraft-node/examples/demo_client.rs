use std::env;
use raft_core::proto::vector_service_client::VectorServiceClient;
use raft_core::proto::{InsertRequest, SearchRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Default ports: Leader on 50051, Follower on 50052
    let leader_port = args.get(1).map(|s| s.as_str()).unwrap_or("50051");
    let follower_port = args.get(2).map(|s| s.as_str()).unwrap_or("50052");

    let leader_addr = format!("http://127.0.0.1:{}", leader_port);
    let follower_addr = format!("http://127.0.0.1:{}", follower_port);

    println!("============================================================");
    println!("  VectoRaft: Distributed Replication & Search Verification");
    println!("  Target Leader   : {}", leader_addr);
    println!("  Target Follower : {}", follower_addr);
    println!("============================================================\n");

    // Connect to the Leader node to insert vectors
    println!("[Step 1] Connecting to Leader at {}...", leader_addr);
    let mut leader_client = VectorServiceClient::connect(leader_addr).await?;

    // Insert Document 42: A vector pointing towards the X axis
    let doc_42 = InsertRequest {
        id: 42,
        values: vec![1.0, 0.0, 0.0],
    };

    println!("[Step 1] Sending vector ID: 42 [1.0, 0.0, 0.0] to Leader...");
    let reply = leader_client.insert(doc_42).await?.into_inner();
    println!(" -> Leader response: success={}, message='{}'\n", reply.success, reply.message);

    // Insert Document 99: A vector pointing towards the Y axis
    let doc_99 = InsertRequest {
        id: 99,
        values: vec![0.0, 1.0, 0.0],
    };

    println!("[Step 1] Sending vector ID: 99 [0.0, 1.0, 0.0] to Leader...");
    let reply = leader_client.insert(doc_99).await?.into_inner();
    println!(" -> Leader response: success={}, message='{}'\n", reply.success, reply.message);

    // Query the FOLLOWER node (which was never directly contacted by the client!)
    println!("[Step 2] Connecting to Follower at {} to perform SIMD search...", follower_addr);
    let mut follower_client = VectorServiceClient::connect(follower_addr).await?;

    // Query looking for something pointing towards X (should match doc 42 perfectly)
    let search_req = SearchRequest {
        query_vector: vec![1.0, 0.0, 0.0],
        top_k: 2,
    };

    println!("[Step 2] Searching on Follower for query [1.0, 0.0, 0.0] (top_k: 2)...");
    let search_reply = follower_client.search(search_req).await?.into_inner();

    println!("\n=== Search Results on Follower ===");
    for (rank, hit) in search_reply.hits.iter().enumerate() {
        println!(
            "Rank {}: Document ID = {}, Cosine Similarity = {:.4}",
            rank + 1, hit.id, hit.score
        );
    }

    assert!(!search_reply.hits.is_empty(), "Follower returned no results!");
    assert_eq!(search_reply.hits[0].id, 42, "Follower failed to find replicated vector 42 as top match!");
    assert!((search_reply.hits[0].score - 1.0).abs() < 1e-5);
    Ok(())
}