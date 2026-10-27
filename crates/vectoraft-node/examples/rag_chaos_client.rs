use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use raft_core::proto::vector_service_client::VectorServiceClient;
use raft_core::proto::{InsertRequest, SearchRequest};

/// A Knowledge Base document for our RAG demo.
struct Document {
    id: u64,
    title: &'static str,
    text: &'static str,
    vector: Vec<f32>,
}

/// Smart VectoRaft Client with automatic failover and leader discovery.
struct SmartVectoRaftClient {
    node_addresses: Vec<String>,
    current_leader_idx: usize,
}

impl SmartVectoRaftClient {
    fn new(addresses: Vec<String>) -> Self {
        Self {
            node_addresses: addresses,
            current_leader_idx: 0,
        }
    }

    /// Attempts to insert a vector, automatically failing over to other nodes if a crash occurs.
    async fn insert_with_retry(&mut self, id: u64, values: Vec<f32>) -> Result<(), String> {
        let max_attempts = 15;

        for _ in 0..max_attempts {
            let addr = self.node_addresses[self.current_leader_idx].clone();

            if let Ok(mut client) = VectorServiceClient::connect(addr.clone()).await {
                let req = InsertRequest {
                    id,
                    values: values.clone(),
                };

                if let Ok(res) = client.insert(req).await {
                    let reply = res.into_inner();
                    if reply.success {
                        return Ok(());
                    }
                }
            }

            // If connection failed or node was not leader, rotate to next node in cluster!
            self.current_leader_idx = (self.current_leader_idx + 1) % self.node_addresses.len();
            sleep(Duration::from_millis(100)).await;
        }

        Err("Failed to insert: Cluster unreachable or no leader elected.".to_string())
    }

    /// Queries any available node in the cluster for vector search.
    async fn search_with_failover(&mut self, query: &[f32], top_k: u64) -> Result<(u64, f32), String> {
        for _ in 0..self.node_addresses.len() {
            let addr = self.node_addresses[self.current_leader_idx].clone();

            if let Ok(mut client) = VectorServiceClient::connect(addr.clone()).await {
                let req = SearchRequest {
                    query_vector: query.to_vec(),
                    top_k,
                };

                let start = Instant::now();
                if let Ok(res) = client.search(req).await {
                    let reply = res.into_inner();
                    if let Some(hit) = reply.hits.first() {
                        let latency = start.elapsed();
                        println!(
                            " -> Search answered by {} in {:.2?} (Doc ID: {}, Score: {:.4})",
                            addr, latency, hit.id, hit.score
                        );
                        return Ok((hit.id, hit.score));
                    }
                }
            }

            self.current_leader_idx = (self.current_leader_idx + 1) % self.node_addresses.len();
            sleep(Duration::from_millis(50)).await;
        }

        Err("Cluster search failed: All nodes unreachable.".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("     VectoRaft: High-Availability RAG Demonstration         ");
    println!("============================================================\n");

    let cluster_nodes = vec![
        "http://127.0.0.1:50051".to_string(),
        "http://127.0.0.1:50052".to_string(),
        "http://127.0.0.1:50053".to_string(),
    ];

    let mut client = SmartVectoRaftClient::new(cluster_nodes);

    // Knowledge Base containing 3 enterprise documents (dimension 3)
    let documents = vec![
        Document {
            id: 1,
            title: "Remote Work Policy",
            text: "Employees are allowed up to 3 days of remote work per week upon manager approval.",
            vector: vec![1.0, 0.0, 0.0],
        },
        Document {
            id: 2,
            title: "Health & Dental Insurance",
            text: "Health coverage is 100% reimbursed for employees and 70% for dependents.",
            vector: vec![0.0, 1.0, 0.0],
        },
        Document {
            id: 3,
            title: "IT Security Guidelines",
            text: "All code repositories require 2FA and hardware security keys for production access.",
            vector: vec![0.0, 0.0, 1.0],
        },
    ];

    let doc_map: HashMap<u64, &Document> = documents.iter().map(|d| (d.id, d)).collect();

    // Ingest Knowledge Base into the cluster
    println!("[Step 1] Ingesting enterprise knowledge base into VectoRaft cluster...");
    for doc in &documents {
        print!(" -> Replicating document: '{}' (ID: {})... ", doc.title, doc.id);
        client.insert_with_retry(doc.id, doc.vector.clone()).await?;
        println!("COMMITTED");
    }
    println!("[Step 1] All documents successfully replicated across the cluster!\n");

    // Continuous Query Loop simulating a production chatbot
    println!("============================================================");
    println!(" [Step 2] Continuous RAG Query Loop Running...             ");
    println!(" >>> PROCEED TO KILL THE LEADER NODE NOW (Ctrl+C in terminal) <<<");
    println!("============================================================\n");

    let mut query_count = 1;

    loop {
        // Alternating questions: Question 1 matches Remote Work, Question 2 matches IT Security
        let (prompt, query_vec) = if query_count % 2 == 1 {
            ("Where can I work during the week?", vec![0.95, 0.05, 0.0])
        } else {
            ("How do I access production servers securely?", vec![0.05, 0.0, 0.95])
        };

        println!("[Query #{}] User asks: '{}'", query_count, prompt);

        match client.search_with_failover(&query_vec, 1).await {
            Ok((doc_id, score)) => {
                if let Some(doc) = doc_map.get(&doc_id) {
                    println!(" [RAG Context Retrieved] '{}' (Relevance: {:.2}%)", doc.title, score * 100.0);
                    println!(" [LLM Response Simulation] Based on '{}': \"{}\"\n", doc.title, doc.text);
                }
            }
            Err(e) => {
                println!(" [ERROR] {}\n", e);
            }
        }

        query_count += 1;
        sleep(Duration::from_millis(1500)).await;
    }
}