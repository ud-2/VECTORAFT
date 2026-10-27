# VectoRaft

Distributed, fault-tolerant vector database written in *Rust*. Designed for high-availability *Retrieval-Augmented Generation (RAG)* pipelines by combining an in-memory SIMD-accelerated similarity search engine with a custom *Raft consensus* implementation.

---

## Architecture Overview

| Layer | Component | Core Technology | Primary Responsibility | Guarantees & Invariants |
| :--- | :--- | :--- | :--- | :--- |
| Network / Transport | `RaftRpcServer`, `VectorRpcHandler` | Tonic (gRPC), Prost (Protobuf 3), HTTP/2 | Handles inter-node RPCs (`RequestVote`, `AppendEntries`) and public client queries (`Insert`, `Search`). | Multiplexed TCP connections, non-blocking asynchronous request handling. |
| Consensus Engine | `RaftNode`, `RaftState` | Tokio runtime, `mpsc`/`oneshot` channels | Drives election timers, quorum verification, and log consistency checks. | Single leader per term, election safety, split-brain immunity ($\lfloor N/2 \rfloor + 1$ quorum). |
| Replication Log | `RaftLog` | Rust `Vec<LogEntry>`, Sliding-window buffer | Buffers uncommitted vector commands and coordinates linearizable updates. | Log matching property; 1-based logical indexing with physical offset translation. |
| Storage & State Machine | `VectorStore` | Rust `std::arch::x86_64` (AVX2/FMA), Min-Heap (`BinaryHeap`) | Executes L2-normalized dot products and extracts Top-K nearest neighbors. | Strictly deterministic execution ($commit\_index \ge last\_applied$). |
| Persistence & Compaction | Snapshot Subsystem | `serde`, `bincode`, `BufWriter`/`BufReader` | Serializes in-memory vectors to disk (`snapshot_node_X.bin`) and truncates replicated logs. | Bounded memory usage, sub-millisecond cold boot recovery from disk. |

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

| Concept | Definition | Authoritative Reference / Reading |
| :--- | :--- | :--- |
| SIMD (Single Instruction, Multiple Data) | Hardware execution model allowing one processor instruction to perform arithmetic on multiple data points simultaneously using wide CPU registers (e.g., 256-bit AVX2 registers handling eight 32-bit floats per clock cycle). | [Intel Intrinsics Guide (AVX2 / FMA)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) |
| Cosine Similarity vs. Dot Product | Cosine similarity evaluates the angle between vectors independent of magnitude. If vectors are pre-normalized to unit length ($\|v\| = 1$), cosine similarity simplifies mathematically to a dot product, eliminating runtime divisions and square roots. | [Vector Similarity Metrics (Pinecone Guide)](https://www.pinecone.io/learn/vector-similarity/) |
| Raft Consensus Algorithm | A distributed consensus protocol designed for understandability, relying on a decomposed leader election, replicated log management, and safety invariants enforced across election terms. | [In Search of an Understandable Consensus Algorithm (Ongaro & Ousterhout, Stanford 2014)](https://raft.github.io/raft.pdf) |
| State Machine Replication (SMR) | A method for building fault-tolerant services where identical deterministic automata transition through identical states by executing the same sequence of committed log inputs. | [Implementing Fault-Tolerant Services Using The State Machine Approach (Fred B. Schneider)](https://www.cs.cornell.edu/fbs/publications/SMSurvey.pdf) |
| Quorum | The minimum number of operational members required to conduct transactions and make authoritative decisions ($\lfloor N/2 \rfloor + 1$). Prevents split-brain by guaranteeing overlapping majorities. | [Quorums in Distributed Systems (MIT 6.824)](https://pdos.csail.mit.edu/6.824/) |
| Actor Pattern in Rust | A concurrency pattern where isolated tasks manage state exclusively and communicate strictly through asynchronous message passing (`mpsc`/`oneshot`), eliminating shared-memory data races and lock deadlocks. | [Actors with Tokio (Official Tokio Guide)](https://ryhl.io/blog/actors-with-tokio/) |
| Sliding-Window Log Compaction | Replacing historical log sequences up to a committed index with an on-disk snapshot while mapping absolute logical indices to truncated relative physical array buffers via index offsets. | [Raft Paper Section 7: Log Compaction](https://raft.github.io/raft.pdf) |
| Binary Serialization (`bincode`) | Compact, non-self-describing binary encoding designed for Rust data structures. Bypasses text parsing overhead (such as JSON or YAML) by mapping directly to memory-aligned primitive layouts. | [Bincode Documentation (docs.rs)](https://docs.rs/bincode/latest/bincode/) |