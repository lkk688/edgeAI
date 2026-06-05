# 🚀 gputool User Guide

**gputool** is a lightweight, non-sudo utility CLI designed to manage services and networking in restricted or locked-down environments (such as university lab machines or shared server nodes) where you **do not have root (`sudo`) privileges**.

Unlike the main `sjsujetsontool`, which installs system-level packages and systemd services on NVIDIA Jetson dev kits, `gputool` runs completely inside the user space, saving all of its state, logs, and sockets locally under `$HOME/.gputool`.

---

## 🆚 `sjsujetsontool` vs `gputool`

| Feature | `sjsujetsontool` | `gputool` |
|---|---|---|
| **Privileges Required** | `sudo` (Root) | **No Sudo** (User space only) |
| **Primary Platforms** | NVIDIA Jetson | Generic Linux (Jetson, x86 Servers, etc.) |
| **Tailscale Mode** | System kernel interface (`tailscale0` tun card) | User-space networking (Proxy mode) |
| **Installation** | System-wide binaries & systemd services | Static binaries in user home directory |
| **Download Dependencies** | Requires system `curl` / `wget` | Falls back to **Python 3** if curl/wget are missing |

---

## 📥 Installation

Choose the command that fits your system's network tools:

### Option A: Standard (curl / wget installed)
```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh | bash
```

### Option B: Locked-down Lab Machines (No `curl` or `wget`)
If the computer has neither `curl` nor `wget` (or they are blocked), use Python 3 to stream the installer:
```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh').read().decode())" | bash
```

---

## 📋 Command Reference

### Core Commands

* **Show Help Menu:**
  ```bash
  gputool help
  ```
* **Print Version:**
  ```bash
  gputool version
  ```
* **Install/Re-install local script:**
  ```bash
  gputool install
  ```
* **Update script from GitHub:**
  ```bash
  gputool update-script
  ```

---

## 🌐 Userspace Tailscale VPN

Since `gputool` runs without root privileges, it cannot create a virtual network interface card (`tailscale0`). Instead, it uses **user-space networking (SOCKS5/HTTP Proxy)** to routing network traffic through the Headscale VPN.

### 1. Download Static Binaries
Downloads the pre-compiled, standalone Tailscale package for your CPU architecture (AMD64, ARM64, or ARM) and configures the binaries in `~/.gputool/tailscale`:
```bash
gputool tailscale setup
```

### 2. Connect to the VPN
Starts the background `tailscaled` daemon in userspace proxy mode and registers this device on the SJSU Headscale network:
```bash
gputool tailscale up
```
*Note: If another device is already registered under the same hostname, use `gputool tailscale up --force` to override.*

### 3. Check Connection & Proxy Details
Displays the current connection state, your assigned Tailscale IP address, and proxy variables:
```bash
gputool tailscale status
```
Example Output:
```
══════════════════════════════════════════════════
🌐 Userspace Tailscale VPN Status
══════════════════════════════════════════════════
📦 Tailscale Version : 1.68.1
🔧 Daemon PID        : 12845
🔌 Socket Path       : /home/student/.gputool/tailscaled.sock

🟢 Connection State  : Running
   Device Hostname   : lab-machine-01
   Tailscale IPs     : 100.64.0.15
   Connected Peers   : 8

🛡️  User-Space Proxy Configuration:
   • SOCKS5 Proxy    : localhost:1055
   • HTTP Proxy      : localhost:1055
══════════════════════════════════════════════════
```

### 4. Disconnect and Stop Daemon
Logs out of the Headscale server and stops the background tailscaled process:
```bash
gputool tailscale down
```

---

## 🔌 How to Access Peers (Proxy Usage)

Because the userspace client does not have a virtual TUN adapter, outgoing traffic is routed through local proxies running on **port 1055**.

### Web Requests & APIs (curl, python, wget)
To access web interfaces or APIs on other Headscale nodes, prefix or export the HTTP/HTTPS proxy:

* **Single command:**
  ```bash
  curl -x http://localhost:1055 http://<peer-tailscale-ip>:<port>
  ```
* **Export for session:**
  ```bash
  export http_proxy=http://localhost:1055
  export https_proxy=http://localhost:1055
  # Now commands run transparently through the proxy
  curl http://<peer-tailscale-ip>:<port>
  ```

### Secure Shell (SSH)
To SSH into another machine on the Headscale network, route the connection via the SOCKS5 proxy command:
```bash
ssh -o ProxyCommand="nc -X 5 -x localhost:1055 %h %p" username@<peer-tailscale-ip>
```

---

## ⚠️ Key Limitations of Non-Sudo Mode

* **Inbound Access is Transparent:** Other machines on the Headscale network can ping or access ports on this computer directly via its Tailscale IP without any proxy setup.
* **Outbound Access Requires Proxies:** This machine *cannot* ping or access other nodes directly unless configured to use the proxy on `localhost:1055` (as shown in the examples above).
