#!/bin/bash
echo "Resetting lab environment..."
sudo docker container prune -f
sudo rm -rf ~/workspace/*
sudo rm -rf ~/.cache/*
echo "✅ Environment reset complete."