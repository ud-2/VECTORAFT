use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

use crate::node::RaftMessage;
use crate::proto::raft_service_server::RaftService;
use crate::proto::{
    AppendEntriesArgs, AppendEntriesReply, RequestVoteArgs, RequestVoteReply,
};

/// Bridges incoming gRPC network requests to the Raft actor via an mpsc channel.
pub struct RaftRpcServer {
    sender: mpsc::Sender<RaftMessage>,
}

impl RaftRpcServer {
    pub fn new(sender: mpsc::Sender<RaftMessage>) -> Self {
        Self { sender }
    }
}

#[tonic::async_trait]
impl RaftService for RaftRpcServer {
    async fn request_vote(
        &self,
        request: Request<RequestVoteArgs>,
    ) -> Result<Response<RequestVoteReply>, Status> {
        let args = request.into_inner();
        let (response_tx, response_rx) = oneshot::channel();

        let message = RaftMessage::RequestVote { args, response_tx };

        // Forward message to Raft actor loop
        self.sender
            .send(message)
            .await
            .map_err(|_| Status::internal("Raft actor loop is unavailable"))?;

        // Await reply from the actor loop
        let reply = response_rx
            .await
            .map_err(|_| Status::internal("Raft actor failed to respond"))?;

        Ok(Response::new(reply))
    }

    async fn append_entries(
        &self,
        request: Request<AppendEntriesArgs>,
    ) -> Result<Response<AppendEntriesReply>, Status> {
        let args = request.into_inner();
        let (response_tx, response_rx) = oneshot::channel();

        let message = RaftMessage::AppendEntries { args, response_tx };

        self.sender
            .send(message)
            .await
            .map_err(|_| Status::internal("Raft actor loop is unavailable"))?;

        let reply = response_rx
            .await
            .map_err(|_| Status::internal("Raft actor failed to respond"))?;

        Ok(Response::new(reply))
    }
}