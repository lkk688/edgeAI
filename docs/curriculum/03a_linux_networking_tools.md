# 🌐 Introduction to Linux and Networking Tools
**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)


## 🧠 Part 1: Basic Computer Networking Concepts

Before using tools, it's essential to understand the fundamental components of computer networks:

### 📡 Key Concepts

| Concept     | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| IP Address  | Unique identifier for a device on a network (e.g., 192.168.1.10)      |
| Subnet      | Logical subdivision of a network                                      |
| Gateway     | Device that routes traffic to other networks                          |
| DNS         | Domain Name System — translates domain names into IP addresses        |
| MAC Address | Hardware address of a network interface                               |
| DHCP        | Automatically assigns IP addresses on a network                       |
| NAT         | Network Address Translation — allows multiple devices to share one IP |
| Port        | Logical access channel for communication (e.g., port 22 = SSH)        |

### 🔀 Network Types

* **LAN (Local Area Network)**: Limited to a building or campus
* **WAN (Wide Area Network)**: Broader networks like the internet
* **Switch**: Connects devices within LAN
* **Router**: Connects LAN to WAN

---

## 🧩 Part 2: The 5 Layers of Computer Networking (Simplified TCP/IP Model)

Computer networking is typically divided into 5 abstract layers, each with specific responsibilities and headers:

| Layer           | Purpose                                       | Example Protocols        | Header Example            |
| --------------- | --------------------------------------------- | ------------------------ | ------------------------- |
| **Application** | Interface for user applications               | HTTP, SSH, DNS, FTP      | HTTP headers, DNS records |
| **Transport**   | Ensures reliable communication, manages ports | TCP, UDP                 | TCP/UDP header (port #s)  |
| **Network**     | Routing and addressing between networks       | IP, ICMP                 | IP header (IP address)    |
| **Data Link**   | Direct link communication on local network    | Ethernet, Wi-Fi (802.11) | MAC header                |
| **Physical**    | Transmission of bits over physical medium     | Cables, Wi-Fi radios     | Voltage/pulse or RF wave  |

### 🛠️ Linux Implementation of Each Layer

| Layer       | Linux Tools / Files               | Kernel/Driver Components            |
| ----------- | --------------------------------- | ----------------------------------- |
| Application | curl, wget, ssh, scp              | User space tools                    |
| Transport   | ss, netstat, iptables             | TCP/UDP stacks in kernel (net/ipv4) |
| Network     | ip, ifconfig, traceroute, ip rule | IP routing tables, netfilter        |
| Data Link   | ethtool, iw, nmcli                | Network driver, MAC layer           |
| Physical    | rfkill, iwconfig, hciconfig       | Wi-Fi/Bluetooth chip drivers        |

---


## 🧪 Network Discovery and Performance Testing

**Check Interfaces and IP Address**

Install net-tools (Debian/Ubuntu-based)
```bash
#apt install -y net-tools #already installed in the container, This will install: ifconfig, netstat, route, arp, etc.
ifconfig
```

Modern Linux systems prefer ip command from iproute2:
```bash
#apt install -y iproute2 #already installed in the container
ip a
ip addr
ip link
```

### 🌐 Discover Devices on LAN

```bash
nmap -sn 192.168.1.0/24
```

### 📊 Measure Speed and Latency

#### `iperf3` — Network Bandwidth Testing

```bash
# On one device (server):
iperf3 -s

# On another device (client):
iperf3 -c <server-ip>
```

#### `ping` — Latency Test

```bash
#apt install -y iputils-ping
ping google.com
```

#### `speedtest-cli` — Internet Speed Test

```bash
#sudo apt install speedtest-cli #already installed in the container
root@sjsujetson-01:/Developer# speedtest-cli
```

🧰 Part 3: Linux Networking Tools Summary

| Tool            | Purpose                                |
| --------------- | -------------------------------------- |
| `ip`            | IP and interface management            |
| `ping`          | Test connectivity                      |
| `ss`            | Check open ports/sockets               |
| `nmap`          | Network discovery and scanning         |
| `ufw`           | Basic firewall management              |
| `curl/wget`     | Web requests and file download         |
| `nmcli`         | Network connection and Wi-Fi control   |
| `bluetoothctl`  | Bluetooth device scanning and pairing  |
| `iperf3`        | Network throughput measurement         |
| `speedtest-cli` | Measure Internet bandwidth and latency |

---
## 📶 Wi-Fi and Bluetooth Networking

Jetson supports both Wi-Fi and Bluetooth, often via M.2 cards or USB dongles.

### 📡 Wi-Fi Management Tools

```bash
nmcli device wifi list      # Scan for Wi-Fi networks
nmcli device wifi connect <SSID> password <password>
iwconfig                   # View wireless settings (deprecated)
```

### 🔵 Bluetooth Tools

```bash
bluetoothctl               # Interactive Bluetooth manager
rfkill list                # Check if Bluetooth/Wi-Fi are blocked
hciconfig                  # View Bluetooth device configuration
```

---



## 🔬 Part 4: Advanced Network Protocol Analysis

### 📊 Understanding Network Headers

Each layer adds its own header to the data packet. Let's examine how to inspect these headers on Jetson:

#### 🔍 Packet Capture with `tcpdump`

```bash
# Install tcpdump if not available
#root@sjsujetson-01:/Developer# apt install tcpdump #already installed inside the container

# Capture packets on specific interface
root@sjsujetson-01:/Developer# tcpdump -i wlP1p1s0 -n -c 10
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on wlP1p1s0, link-type EN10MB (Ethernet), snapshot length 262144 bytes
.....

# Capture HTTP traffic
root@sjsujetson-01:/Developer# tcpdump -i any port 80 -A
tcpdump: data link type LINUX_SLL2
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
.....

# Capture with detailed headers
root@sjsujetson-01:/Developer# tcpdump -i any -v -n icmp
tcpdump: data link type LINUX_SLL2
tcpdump: listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
```

#### 🌐 Layer-by-Layer Analysis

| Layer | Header Fields | Linux Command to Inspect |
|-------|---------------|---------------------------|
| **Application** | HTTP methods, DNS queries | `curl -v`, `dig`, `nslookup` |
| **Transport** | Source/Dest ports, TCP flags | `ss -tuln`, `netstat -tuln` |
| **Network** | Source/Dest IP, TTL | `ip route`, `traceroute` |
| **Data Link** | MAC addresses, VLAN tags | `ip link`, `ethtool` |
| **Physical** | Signal strength, channel | `iwconfig`, `iw dev wlan0 scan` |

### 🔧 Protocol-Specific Tools

#### DNS Analysis
```bash
#These are already installed inside the container
# apt update
# # Install dig and nslookup (part of dnsutils)
# apt install -y dnsutils
# # Install 'time' command (optional, usually pre-installed)
# apt install -y time

# Query DNS records
dig google.com
nslookup google.com

# Check DNS resolution time
time nslookup google.com

# Use specific DNS server
dig @8.8.8.8 google.com
```

#### TCP Connection Analysis
```bash
# Show TCP connection states
ss -tuln

# Monitor TCP connections in real-time
watch -n 1 'ss -tuln | grep :22'

# Check TCP window scaling
ss -i
```

---

## 🛠️ Part 5: Advanced Linux Network Tools

### 🔍 Network Troubleshooting Arsenal

| Tool        | Purpose                  | Example                   | Installation                                  |
|-------------|--------------------------|----------------------------------|-----------------------------------------------|
| `traceroute`| Trace packet path        | `traceroute google.com`          | `sudo apt install traceroute`                 |
| `mtr`       | Continuous traceroute    | `mtr google.com`                 | `sudo apt install mtr`                        |
| `netstat`   | Network statistics       | `netstat -rn` (routing table)    | `sudo apt install net-tools`                  |
| `lsof`      | List open files/sockets  | `lsof -i :22` (SSH connections)  | `sudo apt install lsof`                       |
| `tcpdump`   | Packet capture           | `sudo tcpdump -i wlan0`          | `sudo apt install tcpdump`                    |
| `wireshark` | GUI packet analyzer      | `sudo wireshark`                 | `sudo apt install wireshark`<br>**+ Add user to group**: `sudo usermod -aG wireshark $USER` |
| `ethtool`   | Ethernet tool            | `ethtool eth0`                   | `sudo apt install ethtool`                    |
| `iw`        | Wireless tools           | `iw dev wlan0 info`              | `sudo apt install iw`                         |

> All these tools are already installed inside the container, run `sjsujetsontool shell` to enter into the container.

Packet Capture and Basic Analysis
```bash
# Terminal 1: Capture all packets across interfaces
root@sjsujetson-01:/Developer# tcpdump -i any -w network_capture.pcap

# Terminal 2: Generate some traffic
ping -c 10 google.com
curl -I https://www.google.com

# Terminal 1: Ctrl+C to stop capture

#CLI analysis
root@sjsujetson-01:/Developer# tcpdump -r network_capture.pcap -n
```
### 📡 Wireless Network Deep Dive

#### Wi-Fi Interface Management
```bash
iw dev

# Detailed wireless info
iw dev wlP1p1s0 info

# Check wireless statistics
cat /proc/net/wireless
```

#### Bluetooth Low Energy (BLE) on Jetson
```bash
# Install Bluetooth tools, already in container
#apt install bluez bluez-tools

# Scan for BLE devices
hcitool lescan

# Get device info
hciconfig hci0

# Monitor Bluetooth traffic
btmon
```

### 🔒 Network Security Tools

#### Port Scanning and Security
```bash
#apt install -y nmap

#Check Open Ports on Jetson
root@sjsujetson-01:/Developer# nmap -sS localhost
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-07-15 01:49 UTC
Nmap scan report for localhost (127.0.0.1)
Host is up (0.000014s latency).
Not shown: 997 closed tcp ports (reset)
PORT    STATE SERVICE
22/tcp  open  ssh
111/tcp open  rpcbind
631/tcp open  ipp

# Comprehensive port scan
nmap -sS -O -sV 192.168.1.1

# Scan for vulnerabilities
nmap --script vuln 192.168.1.1

```

#### Firewall Management (need host sudo)
```bash
# UFW (Uncomplicated Firewall)
sudo apt install -y ufw
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw status verbose
# iptables (advanced)
sudo iptables -L -n -v
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
```

<!-- ### 🛠️ Lab Setup

```bash
# Install required tools
sudo apt update
sudo apt install -y tcpdump wireshark nmap mtr-tiny iperf3 \
                    bluez bluez-tools wireless-tools net-tools \
                    python3-scapy python3-socket

# Add user to wireshark group
sudo usermod -a -G wireshark $USER
``` -->

### 📋 Network Layer Analysis

#### Task 1.1: Packet Capture and Analysis
```bash
# Terminal 1 (inside the container): Start packet capture
root@sjsujetson-01:/Developer# tcpdump -i any -w network_capture.pcap
tcpdump: data link type LINUX_SLL2
tcpdump: listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes

# Terminal 2 (inside the container): Generate traffic
root@sjsujetson-01:/workspace# ping -c 10 google.com
root@sjsujetson-01:/workspace# curl -I https://www.google.com

# Stop capture (Ctrl+C in Terminal 1)
# Analyze captured packets
root@sjsujetson-01:/Developer# tcpdump -r network_capture.pcap -n
```

#### Task 1.2: Layer-by-Layer Inspection
```bash
# Physical layer - Wi-Fi signal (not permitted in Jetson)
# iw dev wlP1p1s0 scan | grep -A 5 -B 5 "signal:"

# Data link layer - MAC addresses
ip link show
arp -a

# Network layer - IP routing
ip route show
traceroute 8.8.8.8 # apt install traceroute

# Transport layer - TCP/UDP ports
ss -tuln
lsof -i #run `sudo lsof -i ` in host shows more data

# Application layer - HTTP headers
curl -v http://httpbin.org/get
```


### 🔗 Additional Resources

- [Linux Network Administrators Guide](https://tldp.org/LDP/nag2/index.html)
- [Wireshark User Guide](https://www.wireshark.org/docs/wsug_html_chunked/)
- [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/)
- [TCP/IP Illustrated Series](https://www.informit.com/series/series_detail.aspx?ser=2296329)
- [Network Programming with Python](https://realpython.com/python-sockets/)

