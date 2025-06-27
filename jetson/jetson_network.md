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

## sjsujetsontool
```bash
$ curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/sjsujetsontool.sh -o ~/Developer/sjsujetsontool.sh
sjsujetson@sjsujetson-01:~/Developer$ chmod +x sjsujetsontool.sh 
sjsujetson@sjsujetson-01:~/Developer$ sudo mv sjsujetsontool.sh /usr/local/bin/sjsujetsontool
```

## Student account and shared folder
```bash
$ sudo mv ./Developer /Developer
sjsujetson@sjsujetson-01:/$ sudo adduser student
sjsujetson01qa
sudo usermod -aG docker student
#Make both folders belong to a shared group (docker is fine if student is in that group):
sjsujetson@sjsujetson-01:/$ sudo chgrp -R docker /Developer
sjsujetson@sjsujetson-01:/$ sudo chmod -R 775 /Developer
student@sjsujetson-01:~$ curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/sjsujetsontool.sh -o ~/.local/bin/sjsujetsontool.sh
```

```bash
ssh-copy-id student@sjsujetson-01.local
```

## NFS Shared folder
```bash
sudo apt update
sudo apt install nfs-common
sudo apt install avahi-daemon libnss-mdns
sjsujetson@sjsujetson-01:/$ sudo mkdir -p /mnt/nfs/shared
sudo chown -R student:docker /mnt/nfs/shared
sudo chmod -R 775 /mnt/nfs/shared
sjsujetsontool mount-nfs #Mounts nfs-server.local:/srv/nfs/shared to /mnt/nfs/shared
```

## SSD Duplicate
```bash
diskutil list              # Confirm disk4 is your Jetson SSD
diskutil unmountDisk /dev/disk4
sudo dd if=/dev/rdisk4 of=/Users/john/Desktop/jetson.img bs=4m status=progress
```

##New Device setup
