use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use clap::Parser;
use tokio::sync::{mpsc, oneshot, RwLock};
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use raft_core::node::{RaftMessage, RaftNode};
use raft_core::proto::raft_service_server::RaftServiceServer;
use raft_core::proto::vector_service_server::{VectorService, VectorServiceServer};
use raft_core::proto::{
    InsertReply, InsertRequest, SearchHit, SearchReply, SearchRequest, VectorCommand,
};
use raft_core::rpc::RaftRpcServer;
use vector_engine::VectorStore;

#[derive(Parser, Debug)]
#[command(author, version, about = "VectoRaft Node Runner")]
struct Args {
    #[arg(long)]
    id: u64,

    #[arg(long)]
    port: u16,

    #[arg(long, default_value = "")]
    peers: String,

    /// Vector dimension for this cluster (default: 3 for tests)
    #[arg(long, default_value_t = 3)]
    dimension: usize,
}

fn parse_peers(peers_str: &str) -> HashMap<u64, String> {
    let mut map = HashMap::new();
    if peers_str.trim().is_empty() {
        return map;
    }

    for item in peers_str.split(',') {
        let parts: Vec<&str> = item.split('=').collect();
        if parts.len() == 2 {
            if let Ok(peer_id) = parts[0].parse::<u64>() {
                map.insert(peer_id, parts[1].to_string());
            }
        }
    }
    map
}

/// Public API Handler implementing the VectorService gRPC specification.
pub struct VectorRpcHandler {
    store: Arc<RwLock<VectorStore>>,
    raft_tx: mpsc::Sender<RaftMessage>,
}

#[tonic::async_trait]
impl VectorService for VectorRpcHandler {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertReply>, Status> {
        let req = request.into_inner();
        let (response_tx, response_rx) = oneshot::channel();

        let command = VectorCommand {
            id: req.id,
            values: req.values,
        };

        // Forward insertion command to Raft leader actor
        let message = RaftMessage::ClientInsert {
            command,
            response_tx,
        };

        if self.raft_tx.send(message).await.is_err() {
            return Err(Status::internal("Raft actor loop is offline"));
        }

        match response_rx.await {
            Ok(Ok(())) => Ok(Response::new(InsertReply {
                success: true,
                message: "Vector replicated and committed across quorum.".to_string(),
                leader_address: "".to_string(),
            })),
            Ok(Err(err)) => Ok(Response::new(InsertReply {
                success: false,
                message: format!("Write failed: {}", err),
                leader_address: "".to_string(),
            })),
            Err(_) => Err(Status::internal("Raft actor dropped client response")),
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchReply>, Status> {
        let req = request.into_inner();

        // Read lock: multiple concurrent searches can execute simultaneously!
        let store_guard = self.store.read().await;
        let results = store_guard.search(&req.query_vector, req.top_k as usize);

        let hits = results
            .into_iter()
            .map(|res| SearchHit {
                id: res.id,
                score: res.score,
            })
            .collect();

        Ok(Response::new(SearchReply { hits }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let peer_map = parse_peers(&args.peers);

    println!("==================================================");
    println!("  Launching VectoRaft Node: ID {}", args.id);
    println!("  Listening address       : 127.0.0.1:{}", args.port);
    println!("  Configured peers count  : {}", peer_map.len());
    println!("  Vector dimension        : {}", args.dimension);
    println!("==================================================");

    let snapshot_dir = std::path::Path::new("snapshots");
    std::fs::create_dir_all(snapshot_dir)?;

    // Snapshot file path specific to this node ID
    let snapshot_path = snapshot_dir.join(format!("snapshot_node_{}.bin", args.id));

    // Storage Recovery: Restore from snapshot if it exists on disk!
    let initial_store = if snapshot_path.exists() {
        println!(
            "[Storage] Existing snapshot detected! Restoring state from '{}'...",
            snapshot_path.display()
        );

        let store = VectorStore::load_from_disk(&snapshot_path)?;

        println!("[Storage] Recovery complete. Node memory restored!");
        store
    } else {
        println!("[Storage] No prior snapshot found. Initializing fresh VectorStore.");
        VectorStore::new(args.dimension)
    };

    let store = Arc::new(RwLock::new(initial_store));

    // Initialize the Raft consensus actor with snapshot persistence enabled
    let (raft_node, message_sender) = RaftNode::new(
        args.id,
        peer_map,
        Arc::clone(&store),
        Some(snapshot_path),
    );

    // Spawn Raft consensus loop in background
    tokio::spawn(async move {
        raft_node.run().await;
    });

    // Set up gRPC servers
    let raft_rpc = RaftRpcServer::new(message_sender.clone());
    let vector_rpc = VectorRpcHandler {
        store,
        raft_tx: message_sender,
    };

    let socket_address: SocketAddr = format!("127.0.0.1:{}", args.port).parse()?;
    println!("[Network] gRPC servers listening on {}", socket_address);

    Server::builder()
        .add_service(RaftServiceServer::new(raft_rpc))
        .add_service(VectorServiceServer::new(vector_rpc))
        .serve(socket_address)
        .await?;

    Ok(())
}