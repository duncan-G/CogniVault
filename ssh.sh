#!/bin/bash

# SSH connection variables
HOST="ubuntu@<IP_ADDRESS>"
PRIV_KEY="./my-access-ssh-key.pem"

# SSH command
ssh -i "$PRIV_KEY" "$HOST"

# # Build the docker image
# sudo docker build -t cognivault-dev -f Dockerfile.dev .

# # Run the docker container
# sudo docker run -it \
#     --name cognivault-dev \
#     --gpus all \
#     --ipc=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     -w /app \
#     cognivault-dev