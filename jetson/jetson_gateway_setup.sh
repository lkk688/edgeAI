#!/bin/bash
set -e

# === CONFIGURATION ===
WIFI_IFACE="wlp0s20f3"       # Replace with actual Wi-Fi interface
LAN_IFACE="enp0s25"          # Replace with actual Ethernet interface
GATEWAY_IP="192.168.100.1"
DHCP_RANGE_START="192.168.100.50"
DHCP_RANGE_END="192.168.100.150"
NFS_SHARE="/srv/jetson-share"
COCKPIT_PORT=9090

echo "=== [1/8] Updating system ==="
sudo apt update && sudo apt upgrade -y

echo "=== [2/8] Enable IP forwarding ==="
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/99-ip-forward.conf
sudo sysctl -p /etc/sysctl.d/99-ip-forward.conf

echo "=== [3/8] Configure iptables NAT ==="
sudo iptables -t nat -A POSTROUTING -o "$WIFI_IFACE" -j MASQUERADE
sudo iptables -A FORWARD -i "$LAN_IFACE" -j ACCEPT
sudo apt install -y iptables-persistent
sudo netfilter-persistent save

echo "=== [4/8] Static IP and dnsmasq for DHCP ==="
sudo apt install -y dnsmasq net-tools
sudo nmcli con mod "$LAN_IFACE" ipv4.addresses $GATEWAY_IP/24
sudo nmcli con mod "$LAN_IFACE" ipv4.method manual
sudo systemctl restart NetworkManager

# Backup original config
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.bak

# New dnsmasq config
cat <<EOF | sudo tee /etc/dnsmasq.conf
interface=$LAN_IFACE
dhcp-range=$DHCP_RANGE_START,$DHCP_RANGE_END,12h
domain-needed
bogus-priv
EOF

sudo systemctl restart dnsmasq

echo "=== [5/8] Setup NFS server ==="
sudo apt install -y nfs-kernel-server
sudo mkdir -p $NFS_SHARE/models $NFS_SHARE/docker $NFS_SHARE/logs
sudo chown nobody:nogroup $NFS_SHARE
echo "$NFS_SHARE 192.168.100.0/24(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

echo "=== [6/8] Install Cockpit (Dashboard) ==="
sudo apt install -y cockpit cockpit-dashboard cockpit-machines
sudo systemctl enable --now cockpit.socket
echo "Cockpit will be available at: https://$GATEWAY_IP:$COCKPIT_PORT"

echo "=== [7/8] Enable SSH and Wake-on-LAN tools ==="
sudo apt install -y openssh-server wakeonlan
sudo systemctl enable ssh --now

echo "=== [8/8] Optional: Install VNC access ==="
sudo apt install -y tigervnc-standalone-server xfce4 xfce4-goodies
mkdir -p ~/.vnc
cat <<EOF > ~/.vnc/xstartup
#!/bin/sh
startxfce4 &
EOF
chmod +x ~/.vnc/xstartup

echo "=== DONE: Gateway setup complete! ==="
echo "📡 Jetsons should connect via Ethernet and receive IPs like $DHCP_RANGE_START+"
echo "🌐 Internet access shared from $WIFI_IFACE to $LAN_IFACE"
echo "📁 NFS shared folder available at: $NFS_SHARE"
echo "🖥️ Cockpit web UI: https://$GATEWAY_IP:$COCKPIT_PORT"