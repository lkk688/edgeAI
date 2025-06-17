#!/bin/bash
set -e

# === CONFIGURATION ===
GATEWAY_IP="192.168.100.1"
NFS_MOUNT="/mnt/shared"
MODEL_CACHE="$NFS_MOUNT/models"
DOCKER_IMAGE_PATH="$NFS_MOUNT/docker"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:24.02-py3"
GATEWAY_SSH_PUBKEY="ssh-rsa AAAAB3...your_gateway_public_key_here"

echo "=== [1/6] Update and install essentials ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y nfs-common docker.io curl

echo "=== [2/6] Mount NFS shared folder ==="
sudo mkdir -p $NFS_MOUNT
echo "$GATEWAY_IP:/srv/jetson-share $NFS_MOUNT nfs defaults 0 0" | sudo tee -a /etc/fstab
sudo mount -a

echo "=== [3/6] Set AI model cache directories to NFS ==="
echo "export TORCH_HOME=$MODEL_CACHE/torch" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$MODEL_CACHE/hf" >> ~/.bashrc
source ~/.bashrc

echo "=== [4/6] Setup Docker access ==="
sudo usermod -aG docker $USER
newgrp docker

# Optional: use a preloaded container from NFS if offline
if [ -f "$DOCKER_IMAGE_PATH/pytorch_container.tar" ]; then
  echo "Loading PyTorch container from NFS..."
  sudo docker load -i "$DOCKER_IMAGE_PATH/pytorch_container.tar"
else
  echo "Pulling PyTorch container from NGC..."
  sudo docker pull $PYTORCH_IMAGE
fi

echo "alias lab='docker run --rm -it --runtime nvidia --network host -v \$HOME:/workspace -v $MODEL_CACHE:/models $PYTORCH_IMAGE'" >> ~/.bashrc

echo "=== [5/6] Enable SSH and add gateway key ==="
sudo systemctl enable ssh --now

mkdir -p ~/.ssh
echo "$GATEWAY_SSH_PUBKEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "=== [6/6] Optional: Enable Wake-on-LAN ==="
sudo apt install -y ethtool
IFACE=$(ip route | grep default | awk '{print $5}')
sudo ethtool -s "$IFACE" wol g

echo "✅ Jetson setup complete! Please reboot or re-login."