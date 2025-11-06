# CogniVault gRPC Server

A Python gRPC server containerized with Docker using NVIDIA PyTorch base image.

## Project Structure

```
CogniVault/
├── proto/
│   └── cognivault.proto      # Protocol buffer definitions
├── server.py                  # gRPC server implementation
├── watch_server.py            # File watcher that auto-restarts server on changes
├── test_client.py             # Test client for the gRPC server
├── run.sh                     # Build and run script with volume mount
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build configuration
└── README.md                  # This file
```

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate Python code from proto files:
```bash
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/cognivault.proto
```

3. Run the server:
```bash
python server.py --port 50051
```

### Docker Build and Run

#### Quick Start with File Watching (Recommended)

The `run.sh` script builds the image, runs it with source code mounted as a volume, and enables automatic server restart on file changes:

```bash
./run.sh
```

Or with a custom port:
```bash
PORT=50052 ./run.sh
```

This setup:
- Mounts your source code as a volume (edits are reflected immediately)
- Watches for changes to `.py` and `.proto` files
- Automatically restarts the server when files change
- Regenerates proto files when `.proto` files are modified

#### Manual Docker Commands

1. Build the Docker image:
```bash
docker build -t cognivault-server .
```

2. Run the container with volume mount and file watching:
```bash
docker run -p 50051:50051 -v $(pwd):/app -w /app cognivault-server python watch_server.py --port 50051
```

3. Run without file watching (production mode):
```bash
docker run -p 50051:50051 cognivault-server python server.py --port 50051
```

## gRPC Service

The server implements `CogniVaultService` with the following methods:

- `Ping(PingRequest) -> PingResponse`: Simple connectivity test
- `ProcessData(ProcessDataRequest) -> ProcessDataResponse`: Process data with metadata

## Port Configuration

Default port is 50051. You can change it using:
- Command line: `python server.py --port <PORT>`
- Environment variable in Docker: Override CMD in Dockerfile or use docker run

