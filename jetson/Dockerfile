FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

LABEL maintainer="kaikai.liu@sjsu.edu"

ENV DEBIAN_FRONTEND=noninteractive

# Set up workspace
WORKDIR /workspace

# Install packages at runtime instead of build time
RUN echo '#!/bin/bash\n\
apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    net-tools iputils-ping traceroute nmap tcpdump \
    curl wget git unzip \
    ufw fail2ban \
    tmux nano htop \
    openssh-client \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean' > /workspace/install_packages.sh && \
    chmod +x /workspace/install_packages.sh

CMD ["/bin/bash"]