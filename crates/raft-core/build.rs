fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile our proto file into Rust stubs
    tonic_build::configure()
        .compile_protos(&["../../proto/raft.proto"], &["../../proto"])?;
    Ok(())
}