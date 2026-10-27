use crate::proto::{LogEntry, VectorCommand};

/// Represents the in-memory replicated log supporting log compaction (snapshots).
#[derive(Debug, Clone)]
pub struct RaftLog {
    /// Sliding window buffer of log entries.
    
    /// Index 0 is always the sentinel representing the latest snapshot state.
    pub entries: Vec<LogEntry>,
    
    /// Index of the last entry included in the on-disk snapshot.
    pub snapshot_index: u64,
    
    /// Term of the last entry included in the on-disk snapshot.
    pub snapshot_term: u64,
}

impl Default for RaftLog {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftLog {
    /// Creates an empty log with a sentinel at index 0 and term 0.
    pub fn new() -> Self {
        let sentinel = LogEntry {
            index: 0,
            term: 0,
            command: None,
        };

        Self {
            entries: vec![sentinel],
            snapshot_index: 0,
            snapshot_term: 0,
        }
    }

    /// Returns the logical index of the last entry in the log.
    pub fn last_index(&self) -> u64 {
        self.entries.last().map(|e| e.index).unwrap_or(self.snapshot_index)
    }

    /// Returns the term of the last entry in the log.
    pub fn last_term(&self) -> u64 {
        self.entries.last().map(|e| e.term).unwrap_or(self.snapshot_term)
    }

    /// Appends a new vector command to the log and returns its assigned logical index.
    pub fn append(&mut self, term: u64, command: Option<VectorCommand>) -> u64 {
        let new_index = self.last_index() + 1;
        let entry = LogEntry {
            index: new_index,
            term,
            command,
        };
        self.entries.push(entry);
        new_index
    }

    /// Returns a reference to the entry at logical index (if still in memory).
    pub fn get(&self, logical_index: u64) -> Option<&LogEntry> {
        if logical_index < self.snapshot_index {
            // Already compacted and purged from memory!
            return None;
        }

        let physical_offset = (logical_index - self.snapshot_index) as usize;
        self.entries.get(physical_offset)
    }

    /// Discards all log entries up to `up_to_index` to free RAM.
    /// The entry at `up_to_index` becomes the new sentinel at entries[0].
    pub fn compact(&mut self, up_to_index: u64) -> bool {
        if up_to_index <= self.snapshot_index || up_to_index > self.last_index() {
            return false;
        }

        let physical_offset = (up_to_index - self.snapshot_index) as usize;
        if physical_offset >= self.entries.len() {
            return false;
        }

        let new_snapshot_term = self.entries[physical_offset].term;

        // Drain / slice the vector so entries[physical_offset..] becomes the new entries[0..]
        self.entries = self.entries.split_off(physical_offset);
        self.snapshot_index = up_to_index;
        self.snapshot_term = new_snapshot_term;

        // Ensure the sentinel at index 0 has command cleared to free memory
        self.entries[0].command = None;

        true
    }

    /// Validates continuity against prev_log_index/prev_log_term and appends new entries.
    pub fn append_entries_from_leader(
        &mut self,
        prev_log_index: u64,
        prev_log_term: u64,
        new_entries: Vec<LogEntry>,
    ) -> bool {
        if let Some(entry) = self.get(prev_log_index) {
            if entry.term != prev_log_term {
                return false;
            }
        } else if prev_log_index == self.snapshot_index {
            if prev_log_term != self.snapshot_term {
                return false;
            }
        } else {
            return false;
        }

        for entry in new_entries {
            let logical_index = entry.index;
            if logical_index <= self.snapshot_index {
                continue;
            }

            let physical_offset = (logical_index - self.snapshot_index) as usize;
            if physical_offset < self.entries.len() {
                if self.entries[physical_offset].term != entry.term {
                    self.entries.truncate(physical_offset);
                    self.entries.push(entry);
                }
            } else {
                self.entries.push(entry);
            }
        }

        true
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_initialization_sentinel() {
        let log = RaftLog::new();
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
        assert_eq!(log.entries.len(), 1);
    }

    #[test]
    fn test_append_command_increments_index() {
        let mut log = RaftLog::new();
        let cmd = VectorCommand {
            id: 100,
            values: vec![1.0, 2.0, 3.0],
        };

        let assigned_index = log.append(1, Some(cmd));
        assert_eq!(assigned_index, 1);
        assert_eq!(log.last_index(), 1);
        assert_eq!(log.last_term(), 1);
    }

    #[test]
    fn test_reject_when_prev_log_index_missing() {
        let mut log = RaftLog::new();
        // Leader claims prev_log_index is 5, but our log only has index 0
        let success = log.append_entries_from_leader(5, 1, Vec::new());
        assert!(!success);
    }

    #[test]
    fn test_reject_when_term_mismatch() {
        let mut log = RaftLog::new();
        // Append entry at index 1 with term 1
        log.append(1, None);

        // Leader sends an entry assuming index 1 has term 2 (mismatch!)
        let success = log.append_entries_from_leader(1, 2, Vec::new());
        assert!(!success);
    }

    #[test]
    fn test_log_compaction_and_sliding_window() {
        let mut log = RaftLog::new();

        // Append 5 entries
        for i in 1..=5 {
            log.append(1, Some(crate::proto::VectorCommand {
                id: i,
                values: vec![i as f32],
            }));
        }

        assert_eq!(log.last_index(), 5);
        assert_eq!(log.entries.len(), 6); // 1 sentinel + 5 entries

        // Compact up to index 3 (archive entries 1, 2, 3)
        let success = log.compact(3);
        assert!(success);

        // Memory should be freed: only sentinel (3) + entries (4, 5) remain!
        assert_eq!(log.entries.len(), 3);
        assert_eq!(log.snapshot_index, 3);
        assert_eq!(log.last_index(), 5);

        // Accessing purged entries returns None
        assert!(log.get(1).is_none());
        assert!(log.get(2).is_none());

        // Accessing sentinel and recent entries succeeds!
        assert!(log.get(3).is_some());
        assert_eq!(log.get(4).unwrap().index, 4);
        assert_eq!(log.get(5).unwrap().index, 5);
    }
}