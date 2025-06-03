FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

LABEL maintainer="kaikai.liu@sjsu.edu"

ENV DEBIAN_FRONTEND=noninteractive

# First install essential packages without networking tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    curl wget git unzip \
    tmux nano htop \
    openssh-client \
    libgl1-mesa-glx libglib2.0-0 \
    python3-colcon-common-extensions \
    python3-vcstool \
    && apt-get clean

# Install networking tools separately with network=host to avoid iptables issues
# This will be skipped during build but can be installed when container is running
RUN echo "#!/bin/bash\n\
apt-get update && apt-get install -y net-tools iputils-ping traceroute nmap tcpdump ufw fail2ban" > /usr/local/bin/install_network_tools.sh && \
    chmod +x /usr/local/bin/install_network_tools.sh

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r /workspace/requirements.txt

WORKDIR /workspace

# Add a startup script that will install network tools if needed
RUN echo '#!/bin/bash\n\
if [ "$1" = "--install-network-tools" ]; then\n  /usr/local/bin/install_network_tools.sh\nfi\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]