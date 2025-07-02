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

## 🧠 Summary

Understanding computer networking through both theory and hands-on Linux tools enables students to:

* Configure secure remote access
* Monitor and debug edge AI system connectivity
* Integrate Wi-Fi/Bluetooth peripherals
* Measure performance under real-world conditions

---

Next: [Packet Sniffing & Monitoring](01c_packet_sniffing_monitoring.md) →
