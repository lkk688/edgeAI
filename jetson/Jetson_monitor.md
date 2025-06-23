# Jetson Monitoring
Setting up a full Jetson Monitoring Web Dashboard using open-source tools: jetson-stats, Prometheus, and Grafana.


🚀 Goal

Set up a web dashboard accessible at http://<jetson-ip>:3000 showing:
	•	CPU/GPU/memory usage
	•	Temperatures
	•	Power draw
	•	System stats (load, processes)
	•	Jetson-specific data

✅ Step 1: Install jetson-stats (includes jtop)
```bash
sudo apt update
sudo apt install python3-pip -y
sudo pip3 install -U jetson-stats
sudo jtop #show terminal UI
#sudo apt install htop iotop iftop nvtop sysstat
```


To check:

sudo jtop

This gives you a terminal UI — confirm it’s working.

⸻

✅ Step 2: Enable jtop’s Prometheus Exporter

Create exporter config:

sudo jtop --web

This will:
	•	Start a local Prometheus exporter at http://localhost:8000/metrics
	•	Run jtop in background as a service

Test it:

curl http://localhost:8000/metrics

You should see lots of jetson_* metrics.

⸻

✅ Step 3: Install Prometheus

sudo apt install prometheus -y

Edit the config:

sudo nano /etc/prometheus/prometheus.yml

Under scrape_configs:, add:

  - job_name: 'jetson'
    static_configs:
      - targets: ['localhost:8000']

Restart Prometheus:

sudo systemctl restart prometheus

Check it’s working:

curl http://localhost:9090


⸻

✅ Step 4: Install Grafana

sudo apt install -y apt-transport-https software-properties-common
sudo apt install -y grafana
sudo systemctl enable --now grafana-server

Access Grafana at:

http://<jetson-ip>:3000

Default login:
	•	user: admin
	•	pass: admin (you’ll be prompted to change it)

⸻

✅ Step 5: Add Prometheus as a data source in Grafana
	1.	Go to http://<jetson-ip>:3000
	2.	Login → Settings (gear icon) → Data Sources
	3.	Click Add Data Source
	4.	Choose Prometheus
	5.	Set URL to:

http://localhost:9090


	6.	Click Save & Test

⸻

✅ Step 6: Import Jetson Dashboard into Grafana
	1.	Click “+” → Import
	2.	Paste dashboard ID:

12239

Or use: https://grafana.com/grafana/dashboards/12239-jetson-stats-dashboard

	3.	Select Prometheus as the data source
	4.	Click Import

✅ You now have a full GPU-accelerated Jetson system dashboard!

⸻

🧪 Bonus
	•	To make it remote-accessible:
	•	Forward ports: 3000 (Grafana), 9090 (Prometheus)
	•	Or use ngrok or tailscale for secure tunnels
	•	To monitor multiple Jetsons: add more targets to prometheus.yml

