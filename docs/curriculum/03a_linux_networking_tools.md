# 🌐 Introduction to Linux and Networking Tools

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

## 🧪 Network Discovery and Performance Testing

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
ping google.com
```

#### `speedtest-cli` — Internet Speed Test

```bash
sudo apt install speedtest-cli
speedtest-cli
```

---

## 🧰 Part 3: Linux Networking Tools Summary

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

## 🛡️ Practical Examples for Jetson

* ✅ Check Jetson’s IP on the LAN:

  ```bash
  ip a
  ```

* ✅ Enable SSH server:

  ```bash
  sudo systemctl enable ssh
  sudo systemctl start ssh
  ```

* ✅ Scan for all Jetson devices:

  ```bash
  nmap -sn 192.168.1.0/24
  ```

* ✅ Test `.local` hostname mDNS resolution:

  ```bash
  ping jetson-name.local
  ```

* ✅ Scan Wi-Fi access points:

  ```bash
  nmcli device wifi list
  ```

* ✅ Scan nearby Bluetooth devices:

  ```bash
  bluetoothctl
  scan on
  ```

---

## 🧪 Lab Session: Networking with Your Jetson

### 🎯 Objective

Use Linux tools to explore Jetson's network environment, Wi-Fi/Bluetooth devices, and run speed tests.

### 🛠️ Step-by-Step Tutorial

1. **Check Interfaces and IP Address**

   ```bash
   ip a
   ```

2. **Connect to Wi-Fi**

   ```bash
   nmcli device wifi list
   nmcli device wifi connect "<yourSSID>" password "<yourPassword>"
   ```

3. **Ping Your Gateway**

   ```bash
   ping 192.168.1.1
   ```

4. **Scan Nearby Bluetooth Devices**

   ```bash
   bluetoothctl
   scan on
   # Wait and observe
   ```

5. **Measure Internet Speed**

   ```bash
   speedtest-cli
   ```

6. **Discover Other Devices on LAN**

   ```bash
   nmap -sn 192.168.1.0/24
   ```

7. **(Optional) Test Bandwidth with iperf3**

   * Install on 2 devices (Jetson + another PC)
   * Run on Jetson:

     ```bash
     iperf3 -s
     ```
   * On another device:

     ```bash
     iperf3 -c <Jetson-IP>
     ```

### ✅ Deliverables

* Screenshot or output logs of each step
* Submit speedtest and nmap scan results

---

## 🔬 Part 4: Advanced Network Protocol Analysis

### 📊 Understanding Network Headers

Each layer adds its own header to the data packet. Let's examine how to inspect these headers on Jetson:

#### 🔍 Packet Capture with `tcpdump`

```bash
# Install tcpdump if not available
sudo apt update && sudo apt install tcpdump

# Capture packets on specific interface
sudo tcpdump -i wlan0 -n -c 10

# Capture HTTP traffic
sudo tcpdump -i any port 80 -A

# Capture with detailed headers
sudo tcpdump -i any -v -n icmp
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

| Tool | Purpose | Jetson Example |
|------|---------|----------------|
| `traceroute` | Trace packet path | `traceroute google.com` |
| `mtr` | Continuous traceroute | `mtr google.com` |
| `netstat` | Network statistics | `netstat -rn` (routing table) |
| `lsof` | List open files/sockets | `lsof -i :22` (SSH connections) |
| `tcpdump` | Packet capture | `sudo tcpdump -i wlan0` |
| `wireshark` | GUI packet analyzer | `sudo wireshark` |
| `ethtool` | Ethernet tool | `ethtool eth0` |
| `iw` | Wireless tools | `iw dev wlan0 info` |

### 📡 Wireless Network Deep Dive

#### Wi-Fi Interface Management
```bash
# Detailed wireless info
iw dev wlan0 info

# Scan with detailed output
iw dev wlan0 scan | grep -E "SSID|signal|freq"

# Check wireless statistics
cat /proc/net/wireless

# Monitor wireless events
iw event
```

#### Bluetooth Low Energy (BLE) on Jetson
```bash
# Install Bluetooth tools
sudo apt install bluez bluez-tools

# Scan for BLE devices
sudo hcitool lescan

# Get device info
hciconfig hci0

# Monitor Bluetooth traffic
sudo btmon
```

### 🔒 Network Security Tools

#### Port Scanning and Security
```bash
# Comprehensive port scan
nmap -sS -O -sV 192.168.1.1

# Scan for vulnerabilities
nmap --script vuln 192.168.1.1

# Check open ports on Jetson
sudo nmap -sS localhost
```

#### Firewall Management
```bash
# UFW (Uncomplicated Firewall)
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw status verbose

# iptables (advanced)
sudo iptables -L -n -v
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
```

---

## 🌐 Part 6: Jetson-Specific Networking

### 🔌 Hardware Network Interfaces

#### Ethernet Configuration
```bash
# Check Ethernet link status
ethtool eth0

# Set Ethernet speed (if supported)
sudo ethtool -s eth0 speed 1000 duplex full

# View Ethernet statistics
ethtool -S eth0
```

#### USB Network Adapters
```bash
# List USB devices
lsusb

# Check USB network adapters
lsusb | grep -i network

# Monitor USB events
dmesg | grep -i usb
```

### 📶 Wi-Fi Module Management

#### Intel Wi-Fi Cards (common on Jetson)
```bash
# Check Wi-Fi module
lspci | grep -i wireless

# View Wi-Fi driver info
modinfo iwlwifi

# Restart Wi-Fi module
sudo modprobe -r iwlwifi
sudo modprobe iwlwifi
```

#### Network Manager Configuration
```bash
# NetworkManager status
sudo systemctl status NetworkManager

# List connections
nmcli connection show

# Create static IP connection
nmcli connection add type ethernet con-name static-eth ifname eth0 ip4 192.168.1.100/24 gw4 192.168.1.1
```

---

## 🚀 Part 7: Performance Optimization

### 📈 Network Performance Tuning

#### TCP Buffer Optimization
```bash
# Check current TCP settings
sysctl net.core.rmem_max
sysctl net.core.wmem_max

# Optimize for high-bandwidth networks
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### Network Interface Optimization
```bash
# Check interrupt distribution
cat /proc/interrupts | grep eth0

# Enable/disable network features
sudo ethtool -K eth0 gso off
sudo ethtool -K eth0 tso off
```

### 🔄 Load Balancing and Bonding

```bash
# Create bonded interface (if multiple NICs)
sudo modprobe bonding
echo 'alias bond0 bonding' | sudo tee -a /etc/modprobe.d/bonding.conf
```

---

## 🧪 Comprehensive Lab: Network Mastery on Jetson

### 🎯 Lab Objectives

1. **Protocol Analysis**: Capture and analyze network packets
2. **Performance Testing**: Measure and optimize network performance
3. **Security Assessment**: Scan and secure network interfaces
4. **Wireless Management**: Configure and troubleshoot Wi-Fi/Bluetooth
5. **Network Programming**: Create simple network applications

### 🛠️ Lab Setup

```bash
# Install required tools
sudo apt update
sudo apt install -y tcpdump wireshark nmap mtr-tiny iperf3 \
                    bluez bluez-tools wireless-tools net-tools \
                    python3-scapy python3-socket

# Add user to wireshark group
sudo usermod -a -G wireshark $USER
```

### 📋 Exercise 1: Network Layer Analysis

#### Task 1.1: Packet Capture and Analysis
```bash
# Terminal 1: Start packet capture
sudo tcpdump -i any -w network_capture.pcap

# Terminal 2: Generate traffic
ping -c 10 google.com
curl -I https://www.google.com

# Stop capture (Ctrl+C in Terminal 1)
# Analyze captured packets
tcpdump -r network_capture.pcap -n
```

#### Task 1.2: Layer-by-Layer Inspection
```bash
# Physical layer - Wi-Fi signal
iw dev wlan0 scan | grep -A 5 -B 5 "signal:"

# Data link layer - MAC addresses
ip link show
arp -a

# Network layer - IP routing
ip route show
traceroute 8.8.8.8

# Transport layer - TCP/UDP ports
ss -tuln
lsof -i

# Application layer - HTTP headers
curl -v http://httpbin.org/get
```

### 📋 Exercise 2: Performance Benchmarking

#### Task 2.1: Bandwidth Testing
```bash
# Install iperf3 on two devices
# Device 1 (Jetson):
iperf3 -s -p 5001

# Device 2:
iperf3 -c <jetson-ip> -p 5001 -t 30 -i 1

# Test UDP performance
iperf3 -c <jetson-ip> -u -b 100M
```

#### Task 2.2: Latency Analysis
```bash
# Basic ping test
ping -c 100 8.8.8.8 | tail -1

# Continuous monitoring
mtr --report --report-cycles 100 8.8.8.8

# Jitter measurement
ping -c 100 -i 0.1 8.8.8.8 | awk '/time=/ {print $7}' | cut -d'=' -f2
```

### 📋 Exercise 3: Security Assessment

#### Task 3.1: Port Scanning
```bash
# Scan local network
nmap -sn 192.168.1.0/24

# Detailed scan of Jetson
nmap -sS -sV -O localhost

# Check for common vulnerabilities
nmap --script vuln localhost
```

#### Task 3.2: Firewall Configuration
```bash
# Configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw enable
sudo ufw status verbose
```

### 📋 Exercise 4: Wireless Network Programming

#### Task 4.1: Wi-Fi Scanner Script
```python
#!/usr/bin/env python3
# wifi_scanner.py
import subprocess
import json
import re

def scan_wifi():
    """Scan for Wi-Fi networks and return structured data"""
    try:
        result = subprocess.run(['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 
                               'dev', 'wifi'], capture_output=True, text=True)
        networks = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(':')
                if len(parts) >= 3:
                    networks.append({
                        'ssid': parts[0],
                        'signal': parts[1],
                        'security': parts[2]
                    })
        return networks
    except Exception as e:
        print(f"Error scanning Wi-Fi: {e}")
        return []

if __name__ == "__main__":
    networks = scan_wifi()
    print(json.dumps(networks, indent=2))
```

#### Task 4.2: Network Monitor Script
```python
#!/usr/bin/env python3
# network_monitor.py
import psutil
import time
import json

def get_network_stats():
    """Get current network statistics"""
    stats = psutil.net_io_counters(pernic=True)
    return {
        interface: {
            'bytes_sent': stat.bytes_sent,
            'bytes_recv': stat.bytes_recv,
            'packets_sent': stat.packets_sent,
            'packets_recv': stat.packets_recv
        }
        for interface, stat in stats.items()
    }

def monitor_network(duration=60, interval=5):
    """Monitor network usage over time"""
    print(f"Monitoring network for {duration} seconds...")
    start_stats = get_network_stats()
    time.sleep(duration)
    end_stats = get_network_stats()
    
    for interface in start_stats:
        if interface in end_stats:
            sent_diff = end_stats[interface]['bytes_sent'] - start_stats[interface]['bytes_sent']
            recv_diff = end_stats[interface]['bytes_recv'] - start_stats[interface]['bytes_recv']
            
            print(f"\n{interface}:")
            print(f"  Sent: {sent_diff / 1024 / 1024:.2f} MB")
            print(f"  Received: {recv_diff / 1024 / 1024:.2f} MB")
            print(f"  Total: {(sent_diff + recv_diff) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    monitor_network()
```

### 📋 Exercise 5: Bluetooth Device Discovery

```python
#!/usr/bin/env python3
# bluetooth_scanner.py
import subprocess
import json
import time

def scan_bluetooth_devices():
    """Scan for Bluetooth devices"""
    try:
        # Start scanning
        subprocess.run(['bluetoothctl', 'scan', 'on'], 
                      input='\n', text=True, timeout=2)
        
        # Wait for scan
        time.sleep(10)
        
        # Get devices
        result = subprocess.run(['bluetoothctl', 'devices'], 
                               capture_output=True, text=True)
        
        devices = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('Device'):
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    devices.append({
                        'address': parts[1],
                        'name': parts[2]
                    })
        
        return devices
    except Exception as e:
        print(f"Error scanning Bluetooth: {e}")
        return []

if __name__ == "__main__":
    devices = scan_bluetooth_devices()
    print(json.dumps(devices, indent=2))
```

## 📚 Summary and Next Steps

### 🎯 Key Takeaways

1. **Five-Layer Model**: Understanding how data flows through network layers
2. **Linux Network Stack**: Comprehensive knowledge of Linux networking tools
3. **Jetson Networking**: Platform-specific network configuration and optimization
4. **Protocol Analysis**: Ability to capture and analyze network traffic
5. **Performance Optimization**: Techniques for improving network performance
6. **Security Best Practices**: Network security assessment and hardening

### 🚀 Advanced Topics for Further Learning

- **Software-Defined Networking (SDN)**
- **Network Function Virtualization (NFV)**
- **Container Networking (Docker, Kubernetes)**
- **Edge Computing Network Architectures**
- **5G and IoT Networking Protocols**
- **Network Programming with Python/C++**
- **Real-time Network Monitoring and Analytics**

### 🔗 Additional Resources

- [Linux Network Administrators Guide](https://tldp.org/LDP/nag2/index.html)
- [Wireshark User Guide](https://www.wireshark.org/docs/wsug_html_chunked/)
- [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/)
- [TCP/IP Illustrated Series](https://www.informit.com/series/series_detail.aspx?ser=2296329)
- [Network Programming with Python](https://realpython.com/python-sockets/)

