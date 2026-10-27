# VectoRaft

Distributed, fault-tolerant vector database written in *Rust*. Designed for high-availability *Retrieval-Augmented Generation (RAG)* pipelines by combining an in-memory SIMD-accelerated similarity search engine with a custom *Raft consensus* implementation.

---

## Architecture Overview

| Layer | Component | Core Technology | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| Network / Transport | `RaftRpcServer`, `VectorRpcHandler` | Tonic (gRPC), Prost (Protobuf 3), HTTP/2 | Handles inter-node RPCs (`RequestVote`, `AppendEntries`) and public client queries (`Insert`, `Search`). |
| Consensus Engine | `RaftNode`, `RaftState` | Tokio runtime, `mpsc`/`oneshot` channels | Drives election timers, quorum verification, and log consistency checks. |
| Replication Log | `RaftLog` | Rust `Vec<LogEntry>`, Sliding-window buffer | Buffers uncommitted vector commands and coordinates linearizable updates. |
| Storage & State Machine | `VectorStore` | Rust `std::arch::x86_64` (AVX2/FMA), Min-Heap (`BinaryHeap`) | Executes L2-normalized dot products and extracts Top-K nearest neighbors. |
| Persistence & Compaction | Snapshot Subsystem | `serde`, `bincode`, `BufWriter`/`BufReader` | Serializes in-memory vectors to disk (`snapshot_node_X.bin`) and truncates replicated logs. |

---

## Project Structure

```text
vectoraft/
├── Cargo.toml                      # Workspace root definition
├── proto/
│   └── raft.proto                  # Protobuf schemas (Consensus + Vector API)
└── crates/
    ├── vector-engine/              # SIMD math, L2-norm, VectorStore, persistence
    ├── raft-core/                  # Raft state machine, timers, RPC handlers, log
    └── vectoraft-node/             # Main executable, CLI parsing, RAG chaos demo
```

---

## Build and Run

### Prerequisites
* Rust toolchain (stable, 1.75+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* Protocol Buffer compiler: `sudo apt install protobuf-compiler`

### 1. Build
```bash
cargo build --release
```

### 2. Run Benchmarks
```bash
cargo run --release -p vector-engine --example bench_simd
```

### 3. Launch a 3-Node Cluster
Open three separate terminal windows:

```bash
# Terminal 1 (Node 1)
cargo run -p vectoraft-node -- --id 1 --port 50051 --peers "2=http://127.0.0.1:50052,3=http://127.0.0.1:50053"

# Terminal 2 (Node 2)
cargo run -p vectoraft-node -- --id 2 --port 50052 --peers "1=http://127.0.0.1:50051,3=http://127.0.0.1:50053"

# Terminal 3 (Node 3)
cargo run -p vectoraft-node -- --id 3 --port 50053 --peers "1=http://127.0.0.1:50051,2=http://127.0.0.1:50052"
```

### 4. Execute the Resilient RAG Chaos Client
In a fourth terminal window:
```bash
cargo run -p vectoraft-node --example rag_chaos_client
```

Kill the terminal running the current Leader using `Ctrl+C`. Observe the client maintain query continuity via automatic peer failover.

---

## Appendix: Concepts and References

| Concept | Reference / Reading |
| :--- | :--- |
| SIMD (Single Instruction, Multiple Data) | [Intel Intrinsics Guide (AVX2 / FMA)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) |
| Raft Consensus Algorithm | [In Search of an Understandable Consensus Algorithm (Ongaro & Ousterhout, Stanford 2014)](https://raft.github.io/raft.pdf) |
| State Machine Replication (SMR) | [Implementing Fault-Tolerant Services Using The State Machine Approach (Fred B. Schneider)](https://www.cs.cornell.edu/fbs/publications/SMSurvey.pdf) |
| Actor Pattern in Rust | [Actors with Tokio (Official Tokio Guide)](https://ryhl.io/blog/actors-with-tokio/) |
| Sliding-Window Log Compaction | [Raft Paper Section 7: Log Compaction](https://raft.github.io/raft.pdf) |