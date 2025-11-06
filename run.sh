#!/bin/bash
# Build and run CogniVault gRPC server with volume mount and file watching

set -e

IMAGE_NAME="cognivault-server"
CONTAINER_NAME="cognivault-server"
PORT=${PORT:-50052}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo ""
echo "=== Stopping existing container (if any) ==="
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "=== Starting container with volume mount ==="
echo "Source code is mounted as a volume for live editing"
echo "Port mapping: $PORT:50052"
echo ""

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "$PORT:50052" \
  -v "$SCRIPT_DIR:/app" \
  -w /app \
  "$IMAGE_NAME" \
  python watch_server.py --port 50052

echo ""
echo "=== Container started ==="
echo "Container name: $CONTAINER_NAME"
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop container: docker stop $CONTAINER_NAME"
echo ""
echo "Server is watching for file changes..."
docker logs -f "$CONTAINER_NAME"

