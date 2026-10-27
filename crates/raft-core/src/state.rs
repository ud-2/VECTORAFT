use crate::log::RaftLog;
use crate::proto::{
    AppendEntriesArgs, AppendEntriesReply, RequestVoteArgs, RequestVoteReply,
};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NodeRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug)]
pub struct RaftState {
    pub node_id: u64,
    pub current_term: u64,
    pub voted_for: Option<u64>,
    pub role: NodeRole,

    // State Machine Replication fields
    
    pub log: RaftLog,
    
    /// Highest log entry known to be committed across a cluster quorum.
    pub commit_index: u64,

    /// Highest log entry already applied to the local VectorStore.
    pub last_applied: u64,
}

impl RaftState {
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            current_term: 0,
            voted_for: None,
            role: NodeRole::Follower,
            log: RaftLog::new(),
            commit_index: 0,
            last_applied: 0,
        }
    }

    pub fn become_candidate(&mut self) {
        self.role = NodeRole::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.node_id);
    }

    pub fn become_leader(&mut self) {
        assert_eq!(
            self.role,
            NodeRole::Candidate,
            "Only candidates can transition to leader!"
        );
        self.role = NodeRole::Leader;
    }

    pub fn become_follower(&mut self, term: u64) {
        self.role = NodeRole::Follower;
        self.current_term = term;
        self.voted_for = None;
    }

    /// Handles an incoming RequestVote RPC with Raft's log completeness check.
    pub fn handle_request_vote(&mut self, args: &RequestVoteArgs) -> RequestVoteReply {
        // Rule 1: Reject if term is strictly older than ours
        if args.term < self.current_term {
            return RequestVoteReply {
                term: self.current_term,
                vote_granted: false,
            };
        }

        // Rule 2: If candidate has higher term, step down to Follower
        if args.term > self.current_term {
            self.become_follower(args.term);
        }

        // Check if we already cast our vote in this term
        let vote_available = self.voted_for.is_none() || self.voted_for == Some(args.candidate_id);

        // Raft Safety Rule 5.4.1: Is candidate's log at least as up-to-date as ours?
        let my_last_term = self.log.last_term();
        let my_last_index = self.log.last_index();

        let log_is_up_to_date = if args.last_log_term != my_last_term {
            // Compare terms first: higher term wins!
            args.last_log_term > my_last_term
        } else {
            // Terms are identical: longer log wins!
            args.last_log_index >= my_last_index
        };

        if vote_available && log_is_up_to_date {
            self.voted_for = Some(args.candidate_id);
            RequestVoteReply {
                term: self.current_term,
                vote_granted: true,
            }
        } else {
            RequestVoteReply {
                term: self.current_term,
                vote_granted: false,
            }
        }
    }

    /// Handles an incoming AppendEntries RPC, replicating logs and updating commit index.
    pub fn handle_append_entries(&mut self, args: &AppendEntriesArgs) -> AppendEntriesReply {
        // Rule 1: Reject if leader's term is older than ours
        if args.term < self.current_term {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: self.log.last_index(),
            };
        }

        // Rule 2: Acknowledge valid leader and update term if newer
        if args.term > self.current_term || self.role == NodeRole::Candidate {
            self.become_follower(args.term);
        }

        // Rule 3: Verify log continuity and append entries
        let success = self.log.append_entries_from_leader(
            args.prev_log_index,
            args.prev_log_term,
            args.entries.clone(),
        );

        if success {
            // If leader_commit > commit_index, advance commit_index to min(leader_commit, last_index)
            if args.leader_commit > self.commit_index {
                self.commit_index = args.leader_commit.min(self.log.last_index());
            }

            AppendEntriesReply {
                term: self.current_term,
                success: true,
                match_index: self.log.last_index(),
            }
        } else {
            AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: self.log.last_index(),
            }
        }
    }

    /// Applies all newly committed log entries up to commit_index to the given VectorStore.
    /// Returns the number of applied entries.
    pub fn apply_committed_entries(&mut self, store: &mut vector_engine::VectorStore) -> usize {
        let mut applied_count = 0;

        while self.commit_index > self.last_applied {
            self.last_applied += 1;
            if let Some(entry) = self.log.get(self.last_applied) {
                if let Some(command) = &entry.command {
                    store.insert(command.id, command.values.clone());
                    applied_count += 1;
                }
            }
        }

        applied_count
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let node = RaftState::new(1);
        assert_eq!(node.role, NodeRole::Follower);
        assert_eq!(node.current_term, 0);
        assert_eq!(node.voted_for, None);
    }

    #[test]
    fn test_candidate_transition() {
        let mut node = RaftState::new(1);
        node.become_candidate();

        assert_eq!(node.role, NodeRole::Candidate);
        assert_eq!(node.current_term, 1);
        assert_eq!(node.voted_for, Some(1));
    }

    #[test]
    fn test_vote_rejection_for_stale_term() {
        let mut node = RaftState::new(1);
        node.current_term = 2; // Node is already at Term 2

        let args = RequestVoteArgs {
            term: 1, // Stale candidate from Term 1
            candidate_id: 2,
            last_log_index: 0,
            last_log_term: 0,
        };

        let reply = node.handle_request_vote(&args);
        assert!(!reply.vote_granted);
        assert_eq!(reply.term, 2);
    }

    #[test]
    fn test_vote_grant_and_single_vote_per_term() {
        let mut node = RaftState::new(1);

        let candidate_a = RequestVoteArgs {
            term: 1,
            candidate_id: 2,
            last_log_index: 0,
            last_log_term: 0,
        };

        let candidate_b = RequestVoteArgs {
            term: 1,
            candidate_id: 3,
            last_log_index: 0,
            last_log_term: 0,
        };

        // First vote request from Candidate 2: granted!
        let reply_a = node.handle_request_vote(&candidate_a);
        assert!(reply_a.vote_granted);
        assert_eq!(node.voted_for, Some(2));

        // Second vote request in the same term from Candidate 3: rejected!
        let reply_b = node.handle_request_vote(&candidate_b);
        assert!(!reply_b.vote_granted);
    }

    #[test]
    fn test_reject_vote_if_candidate_log_is_behind() {
        let mut node = RaftState::new(1);
        node.current_term = 1;
        // Node 1 has written up to index 5 in term 1
        for _ in 0..5 {
            node.log.append(1, None);
        }

        // Candidate 2 is on Term 2, but only has logs up to index 2!
        let candidate_args = RequestVoteArgs {
            term: 2,
            candidate_id: 2,
            last_log_index: 2, // Behind! (2 < 5)
            last_log_term: 1,
        };

        let reply = node.handle_request_vote(&candidate_args);
        // Vote MUST be refused to avoid losing entries 3, 4, 5!
        assert!(!reply.vote_granted);
    }

    #[test]
    fn test_grant_vote_if_candidate_log_is_longer_or_equal() {
        let mut node = RaftState::new(1);
        node.current_term = 1;
        node.log.append(1, None); // Index 1

        let candidate_args = RequestVoteArgs {
            term: 2,
            candidate_id: 2,
            last_log_index: 2, // More advanced (2 >= 1)
            last_log_term: 1,
        };

        let reply = node.handle_request_vote(&candidate_args);
        assert!(reply.vote_granted);
    }

    #[test]
    fn test_append_entries_replicates_log_and_advances_commit() {
        let mut node = RaftState::new(2);

        let entries = vec![crate::proto::LogEntry {
            index: 1,
            term: 1,
            command: None,
        }];

        let args = AppendEntriesArgs {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries,
            leader_commit: 1,
        };

        let reply = node.handle_append_entries(&args);
        assert!(reply.success);
        assert_eq!(reply.match_index, 1);
        assert_eq!(node.commit_index, 1);
        assert_eq!(node.log.last_index(), 1);
    }

    #[test]
    fn test_apply_committed_entries_to_vector_store() {
        use vector_engine::VectorStore;

        let mut node = RaftState::new(1);
        let mut store = VectorStore::new(3);

        // Append 2 vector commands into the log
        let cmd1 = crate::proto::VectorCommand {
            id: 101,
            values: vec![1.0, 0.0, 0.0],
        };
        let cmd2 = crate::proto::VectorCommand {
            id: 102,
            values: vec![0.0, 1.0, 0.0],
        };

        node.log.append(1, Some(cmd1)); // Index 1
        node.log.append(1, Some(cmd2)); // Index 2

        // At this point, entries are in the log but NOT committed yet
        let applied = node.apply_committed_entries(&mut store);
        assert_eq!(applied, 0);
        assert_eq!(node.last_applied, 0);

        // Quorum validates entries up to index 2!
        node.commit_index = 2;

        // Now apply loop should process both entries into the VectorStore
        let applied = node.apply_committed_entries(&mut store);
        assert_eq!(applied, 2);
        assert_eq!(node.last_applied, 2);

        // Search the VectorStore to prove the vectors are really there!
        let results = store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 101);
    }
}