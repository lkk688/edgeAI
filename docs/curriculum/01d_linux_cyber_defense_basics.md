# 🛡️ 01d: Linux Cyber Defense Basics

This module introduces fundamental cyber defense concepts in Linux, covering tools and techniques to monitor, secure, and harden systems against common cyber threats. It is aimed at students using Jetson devices but also applies to any Linux-based system.

---

## 🎯 Learning Objectives

* Understand key cyber defense strategies on Linux systems
* Learn how to use essential tools: `ufw`, `iptables`, `auditd`, `fail2ban`, `chkrootkit`, etc.
* Practice defensive monitoring and system hardening

---

## 🧱 Basic Security Concepts

| Concept                      | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| Principle of Least Privilege | Only grant minimal necessary access to users/programs                          |
| System Hardening             | Reduce attack surface by disabling unused services, setting strong permissions |
| Logging & Auditing           | Track changes, log events, monitor system activities                           |
| Real-Time Protection         | Monitor services, detect intrusions, block malicious behavior                  |

---

## 🔧 Core Linux Defense Tools

ufw, fail2ban, and auditd (which require host-level access or systemd)

### 🔒 1. `ufw` – Uncomplicated Firewall

```bash
sudo apt install ufw
sudo ufw enable
sudo ufw allow ssh
sudo ufw status verbose
```

### 🔥 2. `iptables` – Advanced Packet Filtering (optional, advanced)

```bash
sudo iptables -L
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

### 📜 3. `auditd` – Audit Daemon

```bash
sudo apt install auditd audispd-plugins
sudo systemctl start auditd
sudo auditctl -w /etc/passwd -p war -k passwd_monitor
sudo ausearch -k passwd_monitor
```

### 🚫 4. `fail2ban` – Prevent Brute Force Logins

```bash
sudo apt install fail2ban
sudo systemctl start fail2ban
sudo systemctl status fail2ban
```

* Config file: `/etc/fail2ban/jail.conf`

### 🧪 5. `chkrootkit` – Rootkit Scanner

```bash
sudo apt install chkrootkit
sudo chkrootkit
```

### 🧠 6. `clamav` – Open-Source Antivirus

```bash
sudo apt install clamav
sudo freshclam  # update definitions
sudo clamscan -r /home
```

---

## 🔍 System Monitoring Tools

| Tool            | Use                           |
| --------------- | ----------------------------- |
| `top`, `htop`   | View running processes        |
| `netstat`, `ss` | View open network connections |
| `who`, `w`      | Who is logged in              |
| `journalctl`    | System logs                   |
| `ps aux`        | View all running processes    |

---

## 📦 Lab: Harden and Monitor a Jetson Device

### Objective:

Set up basic cyber defense on a Jetson Linux device and test protections.

### Step-by-Step:

1. **Update and Harden:**

```bash
sudo apt update && sudo apt upgrade
sudo ufw enable && sudo ufw allow ssh
```

2. **Install and Configure `fail2ban`:**

```bash
sudo apt install fail2ban
```

3. **Simulate SSH brute force attack (from another machine):**

```bash
hydra -l pi -P /usr/share/wordlists/rockyou.txt ssh://<JETSON-IP>
```

Monitor ban via:

```bash
sudo fail2ban-client status sshd
```

4. **Monitor System Logs:**

```bash
journalctl -xe
```

5. **Run `chkrootkit`:**

```bash
sudo chkrootkit
```

6. **Enable Auditing:**

```bash
sudo auditctl -w /etc/shadow -p wa -k shadow_watch
```

---

## 🚨 Best Practices Recap

* Keep system updated regularly
* Disable unused services and ports
* Use firewalls (`ufw`/`iptables`)
* Log and monitor all activity
* Use strong, unique passwords
* Never run user applications as root

---

## 🧠 Challenge for Students

💡 Try hardening your Jetson and writing a bash script that installs and configures `ufw`, `fail2ban`, `auditd`, and runs a periodic system check.

🏁 Bonus: Set up email alerts for intrusion logs.

---

Next: Move into simulated attacks and red-team tools to learn how to defend in real-world scenarios.
