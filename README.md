# edgeAI


# 🚀 Jetson Lab Automation: Central Gateway & Scalable Provisioning

This repository documents how to set up and manage a lab of NVIDIA Jetson Orin Nano devices using:

- ✅ A **central Ubuntu gateway** (NAT + DHCP + NFS + Dashboard)
- ✅ Automated **Jetson client setup**
- ✅ Centralized **status monitoring**, **remote access**, and **reset hooks**
- ✅ Fully automated using **Ansible**

## 🖥️ 1. Gateway Server Setup (Ubuntu 22.04)

The gateway:
- Shares Wi-Fi internet to Jetsons over Ethernet
- Provides DHCP & DNS
- Shares NFS storage (`/srv/jetson-share`)
- Hosts Cockpit (web UI), SSH, and optional VNC


This guide configures a Ubuntu server to act as the central **gateway and controller** for all Jetson Orin Nano lab devices.

🧱 System Requirements

- Ubuntu 22.04 LTS
- 2 Network Interfaces:
  - Wi-Fi (internet uplink)
  - Ethernet (Jetson LAN via switch)

🔧 Setup Script

Save and run this script as `jetson_gateway_setup.sh`:

```bash
#!/bin/bash
set -e

WIFI_IFACE="wlp0s20f3"       # Wi-Fi (change if needed)
LAN_IFACE="enp0s25"          # Ethernet (connected to Jetsons)
GATEWAY_IP="192.168.100.1"
DHCP_RANGE_START="192.168.100.50"
DHCP_RANGE_END="192.168.100.150"
NFS_SHARE="/srv/jetson-share"
COCKPIT_PORT=9090

# System Update
sudo apt update && sudo apt upgrade -y

# Enable IP Forwarding
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/99-ip-forward.conf
sudo sysctl -p /etc/sysctl.d/99-ip-forward.conf

# NAT Configuration
sudo iptables -t nat -A POSTROUTING -o "$WIFI_IFACE" -j MASQUERADE
sudo iptables -A FORWARD -i "$LAN_IFACE" -j ACCEPT
sudo apt install -y iptables-persistent
sudo netfilter-persistent save

# Static IP and dnsmasq
sudo apt install -y dnsmasq net-tools
sudo nmcli con mod "$LAN_IFACE" ipv4.addresses $GATEWAY_IP/24
sudo nmcli con mod "$LAN_IFACE" ipv4.method manual
sudo systemctl restart NetworkManager
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.bak

cat <<EOF | sudo tee /etc/dnsmasq.conf
interface=$LAN_IFACE
dhcp-range=$DHCP_RANGE_START,$DHCP_RANGE_END,12h
domain-needed
bogus-priv
EOF

sudo systemctl restart dnsmasq

# NFS Server
sudo apt install -y nfs-kernel-server
sudo mkdir -p $NFS_SHARE/models $NFS_SHARE/docker $NFS_SHARE/logs
sudo chown nobody:nogroup $NFS_SHARE
echo "$NFS_SHARE 192.168.100.0/24(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# Cockpit Dashboard
sudo apt install -y cockpit cockpit-dashboard cockpit-machines
sudo systemctl enable --now cockpit.socket

# SSH and Wake-on-LAN
sudo apt install -y openssh-server wakeonlan
sudo systemctl enable ssh --now

# VNC (Optional)
sudo apt install -y tigervnc-standalone-server xfce4 xfce4-goodies
mkdir -p ~/.vnc
echo -e "#!/bin/sh\nstartxfce4 &" > ~/.vnc/xstartup
chmod +x ~/.vnc/xstartup

echo "✅ Gateway setup complete!"
echo "🌐 Cockpit: https://$GATEWAY_IP:$COCKPIT_PORT"
```

---

## 🤖 Jetson Device Setup (Each Jetson)
Each Jetson automatically:
	•	Mounts shared NFS folders
	•	Sets AI model cache to NFS
	•	Installs Docker and a PyTorch container
	•	Configures SSH remote access from gateway
	•	Enables Wake-on-LAN

Save and run this as `jetson_lab_client_setup.sh`:

```bash
#!/bin/bash
set -e

GATEWAY_IP="192.168.100.1"
NFS_MOUNT="/mnt/shared"
MODEL_CACHE="$NFS_MOUNT/models"
DOCKER_IMAGE_PATH="$NFS_MOUNT/docker"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:24.02-py3"
GATEWAY_SSH_PUBKEY="ssh-rsa AAAAB3...replace_this_key..."

# Essentials
sudo apt update && sudo apt upgrade -y
sudo apt install -y nfs-common docker.io curl

# Mount NFS
sudo mkdir -p $NFS_MOUNT
echo "$GATEWAY_IP:/srv/jetson-share $NFS_MOUNT nfs defaults 0 0" | sudo tee -a /etc/fstab
sudo mount -a

# Model cache to NFS
echo "export TORCH_HOME=$MODEL_CACHE/torch" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$MODEL_CACHE/hf" >> ~/.bashrc
source ~/.bashrc

# Docker config
sudo usermod -aG docker $USER
newgrp docker

if [ -f "$DOCKER_IMAGE_PATH/pytorch_container.tar" ]; then
  echo "Loading PyTorch container from local share..."
  sudo docker load -i "$DOCKER_IMAGE_PATH/pytorch_container.tar"
else
  echo "Pulling PyTorch container from NGC..."
  sudo docker pull $PYTORCH_IMAGE
fi

echo "alias lab='docker run --rm -it --runtime nvidia --network host -v \$HOME:/workspace -v $MODEL_CACHE:/models $PYTORCH_IMAGE'" >> ~/.bashrc

# Enable SSH and copy admin key
sudo systemctl enable ssh --now
mkdir -p ~/.ssh
echo "$GATEWAY_SSH_PUBKEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Enable Wake-on-LAN
sudo apt install -y ethtool
IFACE=$(ip route | grep default | awk '{print $5}')
sudo ethtool -s "$IFACE" wol g

echo "✅ Jetson setup complete. Please reboot."
```

# CPU and Memory

Jupyter notebook sample code (also available on Google Colab)
* [C GCC Compiler and Assenbly.ipynb](/cpumemory/C_GCC_Compiler_and_Assenbly.ipynb), [colab link](https://colab.research.google.com/drive/1S7dEt_c4RXU-iKunZmAEYb9A_sZcyq7e?usp=sharing)
* [C Review and Memory Layout.ipynb](/cpumemory/C_Review_and_Memory_Layout.ipynb), [colab link](https://colab.research.google.com/drive/1NkU7XPSIwvwlpsXU3p8zBcdBP4XZgNGk?usp=sharing)
* [CPU System.ipynb](/cpumemory/CPU_System.ipynb), [colab link](https://colab.research.google.com/drive/178iJ4B-Qj8NcRiriNPObpQW3DGDOPSlj?usp=sharing)
* [Memory Mountain.ipynb](/cpumemory/memorymountainv2.ipynb), [colab link](https://colab.research.google.com/drive/14F7NXa3bzeYqK0cIkikLtlkmEDncP-je?usp=sharing)
* [matrixmultiple.ipynb](/cpumemory/matrixmultiple.ipynb), [colab link](https://colab.research.google.com/drive/1qQhOGBLSOZJfCGRjo0lNQYMvQMvgujTx?usp=sharing)
    * different matrix multiple versions, BLAS, Intel OneAPI MKL, Intel OneAPI DPC++, Intel OneAPI basekit
* [Multiprocess_and_Concurrent.ipynb](/cpumemory/Multiprocess_and_Concurrent.ipynb), [colab link](https://colab.research.google.com/drive/1gA3KjRGxGlFCQtLcZ1IUCvCH1eF9U9g0?usp=sharing)

# GPU
Jupyter notebook sample code (also available on Google Colab)
* [GPU_and_Cuda_C++.ipynb](/cpumemory/GPU_and_Cuda_C++.ipynb), [colab link](https://colab.research.google.com/drive/1yzRjf8_9TIH4TFO48ooMyvLGHZdvN6cs?usp=sharing)

# Raspberry Pi
Key links
* [Raspberry Pi5](https://www.raspberrypi.com/products/raspberry-pi-5/)
    * Broadcom BCM2712 2.4GHz quad-core 64-bit Arm Cortex-A76 CPU
    * Dual 4Kp60 HDMI
    * LPDDR4X-4267 SDRAM (4GB and 8GB)
    * PCIe 2.0 x1 interface
* [Documentation](https://www.raspberrypi.com/documentation/)

## Raspberry Pi Setup
Configure Raspberry Pi to enable SSH, I2C, SPI and others: [RaspiConfig](https://www.raspberrypi.com/documentation/computers/configuration.html#the-raspi-config-tool)

Interactive pinout diagram: [pinout](https://pinout.xyz)

Check raspberry pi's IP address via "hostname -I"

Install SMB
```bash
sudo apt install samba samba-common-bin smbclient cifs-utils
lkk@raspberrypi:~ $ chmod 0740 Developer/
lkk@raspberrypi:~ $ sudo smbpasswd -a lkk
sudo nano /etc/samba/smb.conf
sudo service samba restart #restart samba if needed
```
At the end of the file, add the following to share the folder, giving the remote user read/write permissions. Replace the <username> placeholder with the username of your primary user account:
```bash
[share]
    path = /home/<username>/Developer
    read only = no
    public = yes
    writable = yes
```
Connect to the SMB folder via "smb://192.168.86.174"

Install the following packages (default is already installed)
```bash
$ sudo apt install python3-smbus
$ sudo apt-get install -y i2c-tools
$ sudo apt-get install python3-dev python3-rpi.gpio
```

Create python virtual environment and jupyterlab
```bash
lkk@raspberrypi:~ $ mkdir mypyvenv
lkk@raspberrypi:~ $ python3 -m venv ./mypyvenv/
lkk@raspberrypi:~ $ source ./mypyvenv/bin/activate
(mypyvenv) lkk@raspberrypi:~ $ pip install RPi.GPIO
pip3 install gpiozero
pip install lgpio
(mypyvenv) lkk@raspberrypi:~ $ pip install jupyterlab
(mypyvenv) lkk@raspberrypi:~ $ jupyter kernelspec list
(mypyvenv) lkk@raspberrypi:~ $ pip install ipykernel
(mypyvenv) lkk@raspberrypi:~ $ ipython kernel install --user --name=mypyvenv
(mypyvenv) lkk@raspberrypi:~ $ jupyter lab --ip='192.168.86.174' --port=8080 --no-browser
```

Setup the Repo:
```bash
lkk@raspberrypi:~ $ mkdir Developer
lkk@raspberrypi:~ $ cd Developer && git clone https://github.com/lkk688/edgeAI.git
```

Visual Studio Code on Raspberry Pi: [link](https://code.visualstudio.com/docs/setup/raspberry-pi). You can adjust the zoom level in VS Code with the View > Appearance > Zoom commands. Install language extensions (e.g., Python) and Jupyter Extension in VSCode.
```bash
sudo apt update
sudo apt install code
code . #open vscode for one folder
```

Enable VSCode remote tunnel to the Raspberry Pi: 1) Click the User icon in the VSCode in Raspberry Pi, turn on "Remote Tunnel Access", login via Github account; 2) Install "Remote Development" extension in the host VSCode. After the extension is installed, you can see the remote tunnel in the host VSCode.

Enable tunnel as a service (https://code.visualstudio.com/docs/remote/tunnels#_how-can-i-ensure-i-keep-my-tunnel-running)
```bash
code tunnel service install
code tunnel service uninstall
```


Serial
```bash
pip install pyserial
(mypyvenv) lkk@raspberrypi:~/Developer/edgeAI $ python -m serial.tools.miniterm

--- Available ports:
---  1: /dev/ttyUSB0         'USB Serial'

pip install ipywidgets
```
Numpy error of "Original error was: libopenblas.so.0: cannot open shared object file"
```bash
sudo apt install libatlas3-base
sudo apt-get install libopenblas-dev
pip3 install numpy
```


Install OpenCV from https://pypi.org/project/opencv-python/#history
```bash
pip install opencv-python==4.7.0.72
import cv2
print(cv2.version)
4.7.0
``` 

```bash
rpicam-hello -t 0
```

Raspberry Pi picamera2 python: https://github.com/raspberrypi/picamera2/tree/main
