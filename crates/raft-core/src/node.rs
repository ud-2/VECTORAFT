use std::collections::HashMap;
use std::path::{PathBuf};
use std::sync::Arc;
use std::time::Duration;
use rand::Rng;
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::time::sleep;

use crate::proto::raft_service_client::RaftServiceClient;
use crate::proto::{
    AppendEntriesArgs, AppendEntriesReply, RequestVoteArgs, RequestVoteReply, VectorCommand,
};
use crate::state::{NodeRole, RaftState};
use vector_engine::VectorStore;

/// Messages sent across internal channels to the Raft actor loop.
pub enum RaftMessage {
    RequestVote {
        args: RequestVoteArgs,
        response_tx: oneshot::Sender<RequestVoteReply>,
    },
    AppendEntries {
        args: AppendEntriesArgs,
        response_tx: oneshot::Sender<AppendEntriesReply>,
    },
    ClientInsert {
        command: VectorCommand,
        response_tx: oneshot::Sender<Result<(), String>>,
    },
}

pub fn get_random_election_timeout() -> Duration {
    let mut rng = rand::thread_rng();
    let timeout_ms = rng.gen_range(150..=300);
    Duration::from_millis(timeout_ms)
}

pub struct RaftNode {
    pub state: RaftState,
    receiver: mpsc::Receiver<RaftMessage>,
    pub peers: HashMap<u64, String>,
    pub store: Arc<RwLock<VectorStore>>,
    pub snapshot_path: Option<PathBuf>,
}

impl RaftNode {
    pub fn new(
        node_id: u64,
        peers: HashMap<u64, String>,
        store: Arc<RwLock<VectorStore>>,
        snapshot_path: Option<PathBuf>,
    ) -> (Self, mpsc::Sender<RaftMessage>) {
        let (sender, receiver) = mpsc::channel(100);
        let node = Self {
            state: RaftState::new(node_id),
            receiver,
            peers,
            store,
            snapshot_path,
        };
        (node, sender)
    }

    pub async fn run(mut self) {
        println!(
            "[Node {}] Starting cluster node. Total cluster size: {}",
            self.state.node_id,
            self.peers.len() + 1
        );

        loop {
            match self.state.role {
                NodeRole::Follower => self.run_follower().await,
                NodeRole::Candidate => self.run_candidate().await,
                NodeRole::Leader => self.run_leader().await,
            }
        }
    }

    async fn run_follower(&mut self) {
        let timeout = get_random_election_timeout();
        let sleep_timer = sleep(timeout);
        tokio::pin!(sleep_timer);

        tokio::select! {
            _ = &mut sleep_timer => {
                println!(
                    "[Node {}] Heartbeat timeout! Transitioning to Candidate.",
                    self.state.node_id
                );
                self.state.become_candidate();
            }

            Some(msg) = self.receiver.recv() => {
                match msg {
                    RaftMessage::RequestVote { args, response_tx } => {
                        let reply = self.state.handle_request_vote(&args);
                        let _ = response_tx.send(reply);
                    }
                    RaftMessage::AppendEntries { args, response_tx } => {
                        let reply = self.state.handle_append_entries(&args);
                        let _ = response_tx.send(reply);

                        // If leader committed entries, apply them to our local VectorStore!
                        if self.state.commit_index > self.state.last_applied {
                            let mut store_guard = self.store.write().await;
                            let count = self.state.apply_committed_entries(&mut *store_guard);
                            if count > 0 {
                                // Persist follower state to disk as well
                                if let Some(ref path) = self.snapshot_path {
                                    if let Ok(()) = store_guard.save_to_disk(path) {
                                        self.state.log.compact(self.state.last_applied);
                                        println!(
                                            "[Storage] Follower saved snapshot to '{}'. Compacted logs up to index {}.",
                                            path.display(), self.state.last_applied
                                        );
                                    }
                                }

                                println!(
                                    "[Node {}] Follower applied {} new entries to VectorStore (last_applied: {})",
                                    self.state.node_id, count, self.state.last_applied
                                );
                            }
                        }
                    }
                    RaftMessage::ClientInsert { response_tx, .. } => {
                        // Followers reject writes and tell client they are not the leader
                        let _ = response_tx.send(Err("NOT_LEADER".to_string()));
                    }
                }
            }
        }
    }

    async fn run_candidate(&mut self) {
        println!(
            "[Node {}] Starting election for Term {}",
            self.state.node_id, self.state.current_term
        );

        let term = self.state.current_term;
        let candidate_id = self.state.node_id;
        let total_nodes = self.peers.len() + 1;
        let quorum = (total_nodes / 2) + 1;

        let (vote_tx, mut vote_rx) = mpsc::channel(total_nodes);

        for (peer_id, addr) in &self.peers {
            let vote_tx = vote_tx.clone();
            let addr = addr.clone();
            let target_peer_id = *peer_id;
            let args = RequestVoteArgs {
                term,
                candidate_id,
                last_log_index: self.state.log.last_index(),
                last_log_term: self.state.log.last_term(),
            };

            tokio::spawn(async move {
                if let Ok(mut client) = RaftServiceClient::connect(addr).await {
                    let request = tonic::Request::new(args);
                    if let Ok(response) = client.request_vote(request).await {
                        let reply = response.into_inner();
                        let _ = vote_tx.send((target_peer_id, reply)).await;
                    }
                }
            });
        }

        drop(vote_tx);

        let mut votes_granted = 1;
        let election_timer = sleep(get_random_election_timeout());
        tokio::pin!(election_timer);

        loop {
            tokio::select! {
                Some((_peer_id, reply)) = vote_rx.recv() => {
                    if reply.term > self.state.current_term {
                        self.state.become_follower(reply.term);
                        return;
                    }
                    if reply.vote_granted {
                        votes_granted += 1;
                        if votes_granted >= quorum {
                            println!(
                                "[Node {}] WON ELECTION with {}/{} votes! Becoming Leader for Term {}.",
                                self.state.node_id, votes_granted, total_nodes, self.state.current_term
                            );
                            self.state.become_leader();
                            return;
                        }
                    }
                }

                _ = &mut election_timer => {
                    self.state.become_candidate();
                    return;
                }

                Some(msg) = self.receiver.recv() => {
                    match msg {
                        RaftMessage::RequestVote { args, response_tx } => {
                            let reply = self.state.handle_request_vote(&args);
                            let _ = response_tx.send(reply);
                        }
                        RaftMessage::AppendEntries { args, response_tx } => {
                            let reply = self.state.handle_append_entries(&args);
                            let _ = response_tx.send(reply);
                            if reply.success {
                                return;
                            }
                        }
                        RaftMessage::ClientInsert { response_tx, .. } => {
                            let _ = response_tx.send(Err("NOT_LEADER".to_string()));
                        }
                    }
                }
            }
        }
    }

    async fn run_leader(&mut self) {
        println!(
            "[Node {}] *** I AM THE LEADER (Term {}) ***",
            self.state.node_id, self.state.current_term
        );

        let heartbeat_interval = Duration::from_millis(50);
        let total_nodes = self.peers.len() + 1;
        let quorum = (total_nodes / 2) + 1;

        loop {
            let timer = sleep(heartbeat_interval);
            tokio::pin!(timer);

            tokio::select! {
                _ = &mut timer => {
                    // Send periodic heartbeats
                    self.broadcast_append_entries().await;
                }

                Some(msg) = self.receiver.recv() => {
                    match msg {
                        RaftMessage::ClientInsert { command, response_tx } => {
                            let term = self.state.current_term;
                            let entry_index = self.state.log.append(term, Some(command));

                            println!(
                                "[Node {}] Leader received client write. Logged at index {}. Replicating...",
                                self.state.node_id, entry_index
                            );

                            // Replicate new entry to followers
                            let acks = self.replicate_entry(entry_index).await;

                            // Include leader itself in the count
                            let total_acks = acks + 1;

                            if total_acks >= quorum {
                                self.state.commit_index = entry_index;
                                let mut store_guard = self.store.write().await;
                                self.state.apply_committed_entries(&mut *store_guard);

                                // Persist snapshot to disk and compact RAM log
                                if let Some(ref path) = self.snapshot_path {
                                    if let Ok(()) = store_guard.save_to_disk(path) {
                                        self.state.log.compact(self.state.last_applied);
                                        println!(
                                            "[Storage] Snapshot saved to '{}'. Compacted logs up to index {}.",
                                            path.display(), self.state.last_applied
                                        );
                                    }
                                }

                                println!(
                                    "[Node {}] Entry {} COMMITTED by quorum ({}/{}). Applied to VectorStore!",
                                    self.state.node_id, entry_index, total_acks, total_nodes
                                );
                                let _ = response_tx.send(Ok(()));
                            } else {
                                let _ = response_tx.send(Err("QUORUM_NOT_REACHED".to_string()));
                            }
                        }

                        RaftMessage::RequestVote { args, response_tx } => {
                            let reply = self.state.handle_request_vote(&args);
                            let _ = response_tx.send(reply);
                            if reply.term > self.state.current_term {
                                return;
                            }
                        }

                        RaftMessage::AppendEntries { args, response_tx } => {
                            let reply = self.state.handle_append_entries(&args);
                            let _ = response_tx.send(reply);
                            if reply.term > self.state.current_term {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Broadcasts empty heartbeats to all peers.
    async fn broadcast_append_entries(&self) {
        let term = self.state.current_term;
        let leader_id = self.state.node_id;
        let leader_commit = self.state.commit_index;
        let prev_log_index = self.state.log.last_index();
        let prev_log_term = self.state.log.last_term();

        for (_peer_id, addr) in &self.peers {
            let addr = addr.clone();
            let args = AppendEntriesArgs {
                term,
                leader_id,
                prev_log_index,
                prev_log_term,
                entries: Vec::new(),
                leader_commit,
            };

            tokio::spawn(async move {
                if let Ok(mut client) = RaftServiceClient::connect(addr).await {
                    let request = tonic::Request::new(args);
                    let _ = client.append_entries(request).await;
                }
            });
        }
    }

    /// Replicates a specific log entry to all peers and counts successful acknowledgements.
    async fn replicate_entry(&self, entry_index: u64) -> usize {
        let entry = match self.state.log.get(entry_index) {
            Some(e) => e.clone(),
            None => return 0,
        };

        let prev_entry = self.state.log.get(entry_index - 1);
        let prev_log_index = prev_entry.map(|e| e.index).unwrap_or(0);
        let prev_log_term = prev_entry.map(|e| e.term).unwrap_or(0);

        let term = self.state.current_term;
        let leader_id = self.state.node_id;
        let leader_commit = self.state.commit_index;

        let (ack_tx, mut ack_rx) = mpsc::channel(self.peers.len());

        for (_peer_id, addr) in &self.peers {
            let ack_tx = ack_tx.clone();
            let addr = addr.clone();
            let args = AppendEntriesArgs {
                term,
                leader_id,
                prev_log_index,
                prev_log_term,
                entries: vec![entry.clone()],
                leader_commit,
            };

            tokio::spawn(async move {
                if let Ok(mut client) = RaftServiceClient::connect(addr).await {
                    let request = tonic::Request::new(args);
                    if let Ok(response) = client.append_entries(request).await {
                        let reply = response.into_inner();
                        if reply.success {
                            let _ = ack_tx.send(true).await;
                            return;
                        }
                    }
                }
                let _ = ack_tx.send(false).await;
            });
        }

        drop(ack_tx);

        let mut acks = 0;
        while let Some(success) = ack_rx.recv().await {
            if success {
                acks += 1;
            }
        }
        acks
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_follower_transitions_to_candidate_on_timeout() {
        let store = Arc::new(RwLock::new(VectorStore::new(3)));
        let (node, _sender) = RaftNode::new(1, HashMap::new(), store, None);

        let handle = tokio::spawn(async move {
            let mut node = node;
            node.run_follower().await;
            node.state
        });

        let end_state = handle.await.unwrap();

        assert_eq!(end_state.role, NodeRole::Candidate);
        assert_eq!(end_state.current_term, 1);
        assert_eq!(end_state.voted_for, Some(1));
    }
}