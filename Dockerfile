# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Generate Python code from proto files
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/cognivault.proto

# Create proto package init file if it doesn't exist
RUN touch proto/__init__.py

# Expose gRPC port and debugpy port
EXPOSE 50051 5678

# Copy watch server
COPY watch_server.py .

# Run the server with file watching by default
# Can override CMD in docker run to use server.py directly
CMD ["python", "watch_server.py", "--port", "50051"]

