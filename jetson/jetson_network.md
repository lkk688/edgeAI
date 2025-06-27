# Enable mDNS support
Enabling .local and mDNS (Multicast DNS) gives you powerful benefits for zero-configuration networking, especially in environments like Jetson development clusters, classrooms, or multiple headless devices.

Zero-Config Hostname Resolution (No IP Needed). You can connect to your Jetson with: `ssh jetson@myjetson.local`

```bash
sudo apt update
sudo apt install avahi-daemon
sjsujetson@sjsujetson-01:~$ sudo systemctl enable avahi-daemon
Synchronizing state of avahi-daemon.service with SysV service script with /lib/systemd/systemd-sysv-install.
Executing: /lib/systemd/systemd-sysv-install enable avahi-daemon
sjsujetson@sjsujetson-01:~$ sudo systemctl start avahi-daemon
sudo reboot
```
You can now ssh into it via `ssh sjsujetson@sjsujetson-01.local`

## Password-less SSH (with SSH Keys)
```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id sjsujetson@sjsujetson-01.local
```