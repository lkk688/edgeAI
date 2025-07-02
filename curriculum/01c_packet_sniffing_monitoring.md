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

## 🧪 Lab Session: Packet Sniffing on Jetson

### 🎯 Objective

Use Linux tools to sniff, inspect, and analyze network traffic from your Jetson device.

### ✅ Setup

Ensure you're connected to a Wi-Fi/Ethernet network.

### 🛠️ Tasks

1. **Identify active interfaces**

```bash
ip a
```

2. **Capture packets using `tcpdump`**

```bash
sudo tcpdump -i wlan0 -c 50 -w capture.pcap
```

3. **Transfer and open `.pcap` in Wireshark**
   Transfer `capture.pcap` to another computer:

```bash
scp capture.pcap user@host.local:/path/to/pcap
```

Open in Wireshark and explore:

* Protocols
* Source/destination IP
* TCP flags
* Use filters like `http`, `dns`, `tcp.port == 22`
* Follow TCP stream to reconstruct data flows

4. **Live monitor with `iftop` or `nethogs`**

```bash
sudo iftop -i wlan0
sudo nethogs wlan0
```

### 📋 Deliverables

* Screenshot or logs from `tcpdump`, `iftop`, or `nethogs`
* Screenshot of Wireshark with filtered results
* List at least 3 types of protocols you observed
* Short write-up: What did you learn about your traffic?

---

## 🧠 Summary

Packet sniffing tools allow students to:

* Understand network protocol behavior
* Debug real-world communication issues
* Analyze traffic for optimization or intrusion detection
* Explore traffic visually with Wireshark

Next: [Linux Cyber Defense Tools](01d_linux_cyber_defense_basics.md) →
