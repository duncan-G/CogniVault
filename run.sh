#!/bin/bash
# Build and run CogniVault gRPC server with volume mount and file watching

set -e

IMAGE_NAME="cognivault-server"
CONTAINER_NAME="cognivault-server"
PORT=${PORT:-50051}
DEBUG=${DEBUG:-true}
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
echo "Port mapping: $PORT:50051"
if [ "$DEBUG" = "true" ]; then
  echo "Debug mode: ENABLED (port 5678)"
fi
echo ""

# Build docker run command
DOCKER_CMD="docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p $PORT:50051"

# Add debug port mapping if debug mode is enabled
if [ "$DEBUG" = "true" ]; then
  DOCKER_CMD="$DOCKER_CMD -p 5678:5678"
fi

DOCKER_CMD="$DOCKER_CMD -v $SCRIPT_DIR:/app \
  -w /app \
  $IMAGE_NAME"

# Add --debug flag if debug mode is enabled
if [ "$DEBUG" = "true" ]; then
  DOCKER_CMD="$DOCKER_CMD python watch_server.py --port 50051 --debug"
else
  DOCKER_CMD="$DOCKER_CMD python watch_server.py --port 50051"
fi

eval $DOCKER_CMD

echo ""
echo "=== Container started ==="
echo "Container name: $CONTAINER_NAME"
echo ""
echo "=== Useful commands ==="
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop container: docker stop $CONTAINER_NAME"
echo "Interactive shell: docker exec -it $CONTAINER_NAME bash"
echo ""
if [ "$DEBUG" = "true" ]; then
  echo "=== Debugging Setup ==="
  echo "1. In VS Code, go to Run and Debug (Ctrl+Shift+D)"
  echo "2. Select 'Python: Debug in Container' configuration"
  echo "3. Press F5 or click the green play button to attach debugger"
  echo ""
fi
echo "=== VS Code Attachment ==="
echo "1. Install 'Dev Containers' extension in VS Code"
echo "2. Press F1 -> 'Dev Containers: Attach to Running Container...'"
echo "3. Select: $CONTAINER_NAME"
echo ""
echo "Server is watching for file changes..."
docker logs -f "$CONTAINER_NAME"

