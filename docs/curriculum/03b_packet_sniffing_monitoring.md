# 🧪 Packet Sniffing & Monitoring on Linux

## 📦 What Is a Network Packet?

A **network packet** is the basic unit of data transferred over a network. Each packet contains two main parts:

* **Header**: Metadata like source/destination IP, protocol, port, and more.
* **Payload**: Actual data (e.g., part of a file, message, or request).

### 🧬 Common Protocols Seen in Packets

| Protocol   | Layer       | Purpose                         |
| ---------- | ----------- | ------------------------------- |
| Ethernet   | Data Link   | Connect devices on a LAN        |
| IP         | Network     | Routing between networks        |
| TCP/UDP    | Transport   | Reliable/fast transport of data |
| HTTP/HTTPS | Application | Web communication               |
| DNS        | Application | Domain name resolution          |
| ICMP       | Network     | Diagnostics (e.g., ping)        |

---

## 🛠️ Tools for Sniffing and Monitoring

### 1. `tcpdump` — CLI Packet Sniffer

A powerful command-line tool to capture and inspect network traffic.

```bash
sudo apt install tcpdump
sudo tcpdump -i <interface>
```

Common usage:

```bash
sudo tcpdump -i wlan0
sudo tcpdump port 80
sudo tcpdump -w packets.pcap
```

### 2. `wireshark` — GUI Packet Analyzer

Wireshark is a powerful graphical tool used to inspect and analyze network packets in detail.

```bash
sudo apt install wireshark
```

You can run it with:

```bash
sudo wireshark
```

#### 🧪 Common Use Cases

* **Live capture**: Choose interface (e.g., wlan0) to monitor live traffic.
* **Filter traffic** using expressions:

  * `http` (HTTP only)
  * `ip.addr == 192.168.1.10`
  * `tcp.port == 22`
* **Follow TCP stream** to reconstruct conversations
* **Inspect packet structure**: Ethernet, IP, TCP, and payload layers

> 🔐 Tip: If permissions issue occurs, ensure your user is in the `wireshark` group:

```bash
sudo usermod -aG wireshark $USER
newgrp wireshark
```

Wireshark can also open `.pcap` files captured via `tcpdump`, allowing post-capture analysis with GUI.

### 3. `iftop` — Real-Time Bandwidth Monitor

Displays top bandwidth-consuming connections.

```bash
sudo apt install iftop
sudo iftop -i wlan0
```

### 4. `nethogs` — Bandwidth by Process

Shows which local processes use network bandwidth.

```bash
sudo apt install nethogs
sudo nethogs wlan0
```

### 5. `netstat` / `ss` — Socket Info

Inspect current open ports and connections.

```bash
ss -tulnp
```

---

## 🔐 Ethical Use and Legal Reminder

* Only sniff packets on **your own network**.
* Avoid collecting sensitive data without permission.
* Use for **learning, debugging, and securing** your systems.

---

## 🧪 Advanced Lab: Jetson Network Analysis & Protocol Understanding

### 🎯 Objectives

1. Use Jetson-optimized tools for network packet analysis
2. Run Wireshark in containerized environment without sudo privileges
3. Understand network protocols through practical packet inspection
4. Implement automated network monitoring solutions

### ✅ Jetson-Specific Setup

#### Prerequisites

```bash
# Update system and install Docker (if not already installed)
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
# Logout and login again for group changes to take effect

# Install additional network tools
sudo apt install tcpdump iftop nethogs nmap tshark -y
```

#### Network Interface Configuration

```bash
# Check available network interfaces
ip link show

# For Jetson devices, common interfaces:
# - eth0: Ethernet
# - wlan0: WiFi (if WiFi module present)
# - docker0: Docker bridge
# - l4tbr0: Jetson-specific bridge

# Enable promiscuous mode for packet capture (if needed)
sudo ip link set wlan0 promisc on
```

### 🐳 Containerized Wireshark Setup (No Sudo Required)

#### Method 1: X11 Forwarding with Docker

```bash
# Create Wireshark container with X11 support
docker run -it --rm \
  --name wireshark-jetson \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/captures:/captures \
  linuxserver/wireshark:latest
```

#### Method 2: Web-based Wireshark

```bash
# Run Wireshark with web interface
docker run -d \
  --name wireshark-web \
  --net=host \
  --privileged \
  -p 3000:3000 \
  -v $(pwd)/captures:/captures \
  -e PUID=1000 \
  -e PGID=1000 \
  linuxserver/wireshark:latest

# Access via browser: http://jetson-ip:3000
```

#### Method 3: Custom Jetson Wireshark Container

```dockerfile
# Dockerfile.wireshark-jetson
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wireshark \
    tshark \
    tcpdump \
    net-tools \
    iputils-ping \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user to wireshark group
RUN groupadd -r wireshark && \
    useradd -r -g wireshark -s /bin/bash wireshark && \
    usermod -a -G wireshark wireshark

# Set capabilities for non-root packet capture
RUN setcap cap_net_raw,cap_net_admin=eip /usr/bin/dumpcap

USER wireshark
WORKDIR /home/wireshark

CMD ["/bin/bash"]
```

```bash
# Build and run custom container
docker build -f Dockerfile.wireshark-jetson -t jetson-wireshark .

docker run -it --rm \
  --name jetson-wireshark \
  --net=host \
  --cap-add=NET_RAW \
  --cap-add=NET_ADMIN \
  -v $(pwd)/captures:/home/wireshark/captures \
  jetson-wireshark
```

### 🛠️ Advanced Packet Analysis Tasks

#### Task 1: Multi-Interface Monitoring

```bash
# Monitor multiple interfaces simultaneously
# Terminal 1: Ethernet traffic
sudo tcpdump -i eth0 -w eth0_capture.pcap &

# Terminal 2: WiFi traffic (if available)
sudo tcpdump -i wlan0 -w wlan0_capture.pcap &

# Terminal 3: Docker bridge traffic
sudo tcpdump -i docker0 -w docker_capture.pcap &

# Let it run for 2-3 minutes, then stop all captures
sudo pkill tcpdump
```

#### Task 2: Protocol-Specific Analysis

```bash
# Capture specific protocols

# 1. HTTP/HTTPS traffic analysis
sudo tcpdump -i any 'port 80 or port 443' -w web_traffic.pcap -c 100

# 2. DNS queries and responses
sudo tcpdump -i any 'port 53' -w dns_traffic.pcap -c 50

# 3. SSH connections
sudo tcpdump -i any 'port 22' -w ssh_traffic.pcap -c 30

# 4. IoT device communication (common ports)
sudo tcpdump -i any 'port 1883 or port 8883 or port 5683' -w iot_traffic.pcap -c 100
```

#### Task 3: Real-time Protocol Analysis with tshark

```bash
# Real-time protocol statistics
tshark -i any -q -z conv,ip -a duration:60

# Real-time HTTP analysis
tshark -i any -Y "http" -T fields -e ip.src -e ip.dst -e http.host -e http.request.uri

# Real-time DNS analysis
tshark -i any -Y "dns" -T fields -e ip.src -e dns.qry.name -e dns.resp.addr
```

### 📊 Network Protocol Understanding Exercises

#### Exercise 1: TCP Three-Way Handshake Analysis

```bash
# Capture TCP handshake
sudo tcpdump -i any 'tcp[tcpflags] & (tcp-syn|tcp-ack) != 0' -w handshake.pcap -c 20

# Generate some TCP connections
curl -s http://httpbin.org/get > /dev/null
wget -q -O /dev/null http://httpbin.org/json
```

**Wireshark Analysis Steps:**
1. Open `handshake.pcap` in Wireshark
2. Filter: `tcp.flags.syn==1`
3. Follow TCP stream for complete handshake
4. Identify: SYN → SYN-ACK → ACK sequence
5. Note sequence numbers and window sizes

#### Exercise 2: HTTP vs HTTPS Traffic Comparison

```bash
# Capture mixed HTTP/HTTPS traffic
sudo tcpdump -i any 'port 80 or port 443' -w http_vs_https.pcap &

# Generate HTTP traffic
curl -s http://httpbin.org/user-agent
curl -s http://httpbin.org/headers

# Generate HTTPS traffic
curl -s https://httpbin.org/user-agent
curl -s https://httpbin.org/headers

sudo pkill tcpdump
```

**Analysis Questions:**
1. Can you read HTTP request headers in plaintext?
2. What do you see in HTTPS packets after the TLS handshake?
3. Compare packet sizes between HTTP and HTTPS
4. Identify TLS version and cipher suites used

#### Exercise 3: DNS Resolution Deep Dive

```bash
# Capture DNS traffic
sudo tcpdump -i any 'port 53' -w dns_analysis.pcap &

# Generate various DNS queries
nslookup google.com
nslookup -type=MX google.com
nslookup -type=AAAA google.com
dig @8.8.8.8 nvidia.com
dig @1.1.1.1 jetson.nvidia.com

sudo pkill tcpdump
```

**Wireshark Analysis:**
1. Filter: `dns`
2. Examine query types (A, AAAA, MX, etc.)
3. Compare response times from different DNS servers
4. Identify recursive vs iterative queries
5. Look for DNS over HTTPS (DoH) traffic on port 443

### 🔍 Advanced Monitoring Scripts

#### Automated Network Monitoring Script

```python
#!/usr/bin/env python3
# jetson_network_monitor.py

import subprocess
import time
import json
import psutil
from datetime import datetime
import threading

class JetsonNetworkMonitor:
    def __init__(self, interface="any", capture_duration=300):
        self.interface = interface
        self.capture_duration = capture_duration
        self.stats = {
            "protocols": {},
            "top_talkers": {},
            "bandwidth_usage": [],
            "start_time": datetime.now().isoformat()
        }
    
    def capture_packets(self, output_file):
        """Capture packets using tcpdump"""
        cmd = [
            "sudo", "tcpdump", "-i", self.interface,
            "-w", output_file,
            "-G", str(self.capture_duration),
            "-W", "1"
        ]
        
        try:
            subprocess.run(cmd, timeout=self.capture_duration + 10)
        except subprocess.TimeoutExpired:
            print("Capture completed")
    
    def analyze_protocols(self, pcap_file):
        """Analyze protocols using tshark"""
        cmd = [
            "tshark", "-r", pcap_file,
            "-q", "-z", "prot,colinfo"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Parse protocol statistics
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('='):
                    parts = line.split()
                    if len(parts) >= 2:
                        protocol = parts[0]
                        count = parts[1]
                        self.stats["protocols"][protocol] = int(count)
        except Exception as e:
            print(f"Protocol analysis error: {e}")
    
    def monitor_bandwidth(self):
        """Monitor real-time bandwidth usage"""
        start_time = time.time()
        
        while time.time() - start_time < self.capture_duration:
            net_io = psutil.net_io_counters()
            timestamp = datetime.now().isoformat()
            
            self.stats["bandwidth_usage"].append({
                "timestamp": timestamp,
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            })
            
            time.sleep(5)  # Sample every 5 seconds
    
    def get_top_talkers(self, pcap_file):
        """Identify top talking hosts"""
        cmd = [
            "tshark", "-r", pcap_file,
            "-q", "-z", "conv,ip"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Parse conversation statistics
            lines = result.stdout.split('\n')
            for line in lines:
                if '<->' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        src_ip = parts[0]
                        dst_ip = parts[2]
                        bytes_total = parts[4]
                        
                        conversation = f"{src_ip} <-> {dst_ip}"
                        self.stats["top_talkers"][conversation] = bytes_total
        except Exception as e:
            print(f"Top talkers analysis error: {e}")
    
    def run_monitoring(self):
        """Run complete monitoring session"""
        print(f"Starting network monitoring on {self.interface} for {self.capture_duration} seconds...")
        
        # Start bandwidth monitoring in background
        bandwidth_thread = threading.Thread(target=self.monitor_bandwidth)
        bandwidth_thread.start()
        
        # Capture packets
        pcap_file = f"jetson_capture_{int(time.time())}.pcap"
        self.capture_packets(pcap_file)
        
        # Wait for bandwidth monitoring to complete
        bandwidth_thread.join()
        
        # Analyze captured data
        print("Analyzing captured packets...")
        self.analyze_protocols(pcap_file)
        self.get_top_talkers(pcap_file)
        
        # Save results
        results_file = f"network_analysis_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Analysis complete. Results saved to {results_file}")
        print(f"Packet capture saved to {pcap_file}")
        
        return self.stats

if __name__ == "__main__":
    monitor = JetsonNetworkMonitor(interface="any", capture_duration=60)
    results = monitor.run_monitoring()
    
    print("\n=== Network Analysis Summary ===")
    print(f"Protocols detected: {list(results['protocols'].keys())}")
    print(f"Total conversations: {len(results['top_talkers'])}")
    print(f"Monitoring duration: {len(results['bandwidth_usage']) * 5} seconds")
```

#### Real-time Protocol Dashboard

```python
#!/usr/bin/env python3
# protocol_dashboard.py

import subprocess
import time
import curses
from collections import defaultdict, deque
import threading

class ProtocolDashboard:
    def __init__(self, interface="any"):
        self.interface = interface
        self.protocol_counts = defaultdict(int)
        self.packet_history = deque(maxlen=100)
        self.running = True
    
    def packet_capture_worker(self):
        """Background packet capture and analysis"""
        cmd = [
            "sudo", "tshark", "-i", self.interface,
            "-T", "fields", "-e", "frame.protocols",
            "-l"  # Line buffered output
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, text=True)
        
        while self.running:
            try:
                line = process.stdout.readline()
                if line:
                    protocols = line.strip().split(':')
                    for protocol in protocols:
                        if protocol:
                            self.protocol_counts[protocol] += 1
                    
                    self.packet_history.append({
                        'timestamp': time.time(),
                        'protocols': protocols
                    })
            except Exception as e:
                break
        
        process.terminate()
    
    def display_dashboard(self, stdscr):
        """Display real-time dashboard using curses"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        # Start packet capture in background
        capture_thread = threading.Thread(target=self.packet_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        
        while True:
            stdscr.clear()
            
            # Header
            stdscr.addstr(0, 0, "Jetson Network Protocol Dashboard", curses.A_BOLD)
            stdscr.addstr(1, 0, f"Interface: {self.interface}", curses.A_UNDERLINE)
            stdscr.addstr(2, 0, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            stdscr.addstr(3, 0, "-" * 60)
            
            # Protocol statistics
            stdscr.addstr(4, 0, "Protocol Statistics:", curses.A_BOLD)
            row = 5
            
            # Sort protocols by count
            sorted_protocols = sorted(self.protocol_counts.items(), 
                                    key=lambda x: x[1], reverse=True)
            
            for protocol, count in sorted_protocols[:15]:  # Top 15
                bar_length = min(40, count // 10) if count > 0 else 0
                bar = "█" * bar_length
                stdscr.addstr(row, 0, f"{protocol:15} {count:6} {bar}")
                row += 1
            
            # Recent activity
            stdscr.addstr(row + 1, 0, "Recent Activity:", curses.A_BOLD)
            recent_packets = list(self.packet_history)[-10:]  # Last 10 packets
            
            for i, packet in enumerate(recent_packets):
                timestamp = time.strftime('%H:%M:%S', 
                                        time.localtime(packet['timestamp']))
                protocols_str = ' -> '.join(packet['protocols'][:5])  # First 5 protocols
                stdscr.addstr(row + 2 + i, 0, f"{timestamp}: {protocols_str}")
            
            stdscr.addstr(row + 13, 0, "Press 'q' to quit", curses.A_DIM)
            stdscr.refresh()
            
            # Check for quit
            key = stdscr.getch()
            if key == ord('q'):
                self.running = False
                break
            
            time.sleep(1)

def main():
    dashboard = ProtocolDashboard()
    curses.wrapper(dashboard.display_dashboard)

if __name__ == "__main__":
    main()
```

### 📋 Comprehensive Lab Deliverables

#### Part 1: Container Setup (20 points)
- [ ] Successfully run Wireshark in Docker container without sudo
- [ ] Capture packets from at least 2 different network interfaces
- [ ] Screenshot of containerized Wireshark interface

#### Part 2: Protocol Analysis (30 points)
- [ ] Analyze TCP three-way handshake with sequence numbers
- [ ] Compare HTTP vs HTTPS packet contents
- [ ] Identify at least 5 different protocols in your captures
- [ ] Document DNS resolution process with timing analysis

#### Part 3: Advanced Monitoring (25 points)
- [ ] Run the automated monitoring script for 5 minutes
- [ ] Generate network analysis JSON report
- [ ] Identify top 3 network conversations
- [ ] Create protocol distribution chart

#### Part 4: Real-time Dashboard (15 points)
- [ ] Run the protocol dashboard for 2 minutes
- [ ] Screenshot showing real-time protocol statistics
- [ ] Document any unusual or interesting traffic patterns

#### Part 5: Security Analysis (10 points)
- [ ] Identify any unencrypted sensitive data in captures
- [ ] Document potential security concerns observed
- [ ] Suggest improvements for network security

### 🎯 Advanced Challenge: IoT Device Analysis

**Scenario**: Your Jetson is connected to a network with various IoT devices.

**Tasks**:
1. Identify IoT devices by their traffic patterns
2. Analyze MQTT, CoAP, or other IoT protocols
3. Create a device fingerprinting report
4. Implement automated anomaly detection

**Bonus**: Integrate with NVIDIA DeepStream for AI-powered network analysis

### 📝 Report Template

```markdown
# Jetson Network Analysis Report

## Executive Summary
- Monitoring duration: X minutes
- Total packets captured: X
- Unique protocols identified: X
- Security findings: X

## Network Overview
- Primary interface: 
- IP configuration:
- Gateway and DNS servers:

## Protocol Analysis
### Most Common Protocols
1. Protocol A (X% of traffic)
2. Protocol B (X% of traffic)
3. Protocol C (X% of traffic)

### Interesting Findings
- Unusual protocols detected:
- Potential security concerns:
- Performance bottlenecks:

## Container Performance
- Wireshark container resource usage:
- Capture performance comparison:
- Recommendations:

## Conclusions and Recommendations
- Network optimization suggestions:
- Security improvements:
- Future monitoring strategies:
```

This comprehensive lab provides hands-on experience with advanced network analysis on Jetson devices, emphasizing containerized tools, protocol understanding, and practical security analysis.

