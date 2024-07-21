# Raspberry Pi
Key links
* [Raspberry Pi5](https://www.raspberrypi.com/products/raspberry-pi-5/)
    * Broadcom BCM2712 2.4GHz quad-core 64-bit Arm Cortex-A76 CPU
    * Dual 4Kp60 HDMI
    * LPDDR4X-4267 SDRAM (4GB and 8GB)
    * PCIe 2.0 x1 interface
* [Documentation](https://www.raspberrypi.com/documentation/)
    * [Remote Access](https://www.raspberrypi.com/documentation/computers/remote-access.html#remote-control-over-the-internet)
* [Raspberry Pi Products](https://www.raspberrypi.com/products/)
* [Canakit Raspberry Pi 5 8GB RAM Starter kit](https://www.canakit.com/canakit-raspberry-pi-5-starter-kit-turbine-black.html)
    * Samsung EVO+ 128GB MicroSD Card pre-loaded with Raspberry Pi OS
    * Set of 2 Micro HDMI to HDMI Cables (6-foot each)
* [Raspberry Pi OS](https://www.raspberrypi.com/documentation/computers/os.html)
    * Raspberry Pi OS is a free, Debian-based operating system optimised for the Raspberry Pi hardware. The latest version of Raspberry Pi OS is based on Debian Bookworm. The previous version was based on Debian Bullseye.
    * apt stores a list of software sources in a file at /etc/apt/sources.list. Before installing software, run the following command `sudo apt update` to update your local list of packages using /etc/apt/sources.list.
    * Run the following command `sudo apt full-upgrade` to upgrade all your installed packages to their latest versions. Unlike Debian, Raspberry Pi OS is under continual development. As a result, package dependencies sometimes change, so you should always use full-upgrade instead of the standard upgrade.
    * To search the archives for a package, pass a search keyword to `apt-cache search <keyword>`, Use the following command to view detailed information about a package: `apt-cache show <package-name>`
    * Install a package: `sudo apt install <package-name>`, Uninstall a package: `sudo apt remove <package-name>`
    * To update the firmware on your Raspberry Pi to the latest version, use `rpi-update`.

## Raspberry Pi Setup
Configure Raspberry Pi to enable SSH, I2C, SPI and others: [RaspiConfig](https://www.raspberrypi.com/documentation/computers/configuration.html#the-raspi-config-tool)

Interactive pinout diagram: [pinout](https://pinout.xyz)

Change the screen scale for high resolution monitors: click Menu > Preferences > Appearance Settings > Defaults tab, then pick a screen size default

Check OS version and know which version of Raspberry Pi OS is running
```bash
$ cat /etc/os-release
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
VERSION_CODENAME=bookworm
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
$ hostnamectl
 Static hostname: raspberrypi
       Icon name: computer
      Machine ID: 5221466cfcc54c08959d13652c2effee
         Boot ID: f9f7fe1972764143bf264d50da30c7c1
Operating System: Debian GNU/Linux 12 (bookworm)  
          Kernel: Linux 6.6.31+rpt-rpi-2712
    Architecture: arm64
#To print Linux distribution-specific information
$ lsb_release -a
No LSB modules are available.
Distributor ID: Debian
Description:    Debian GNU/Linux 12 (bookworm)
Release:        12
Codename:       bookworm
#print Linux or Unix kernel version 
$ uname -mrs
Linux 6.6.31+rpt-rpi-2712 aarch64
#The minor version is:
$ cat /etc/debian_version
12.2
#Use the getconf command check 32bit or 64bit
$ getconf LONG_BIT
64
#see CPU version for your hardware
$ cat /proc/cpuinfo
$ lscpu
#Check architecture
$ dpkg --print-architecture
arm64
$ dpkg --print-foreign-architectures
amrhf
$ sudo dpkg --remove-architecture amrhf
$ dpkg --print-foreign-architectures
```

ARMHF is an architecture used on 32 bit processors. ARM64 is modern version made for 64 bit processors (RPI 5 is the first 64 bit raspberry).
The output of `dpkg --print-architecture`, https://wiki.debian.org/Multiarch/HOWTO
```bash
sudo apt-get update
sudo dpkg --add-architecture amrhf
sudo dpkg --add-architecture aarch64

sudo apt update
dpkg --remove-architecture armhf
dpkg --remove-architecture aarch64
```

## Raspberry Pi Network
Check raspberry pi's IP address via `hostname -I`

You can use the built-in Network Manager CLI (nmcli) to access details about your network. Run the following command: `nmcli device show`

Raspberry Pi OS supports multicast DNS as part of the Avahi service. If your device supports mDNS, you can reach your Raspberry Pi by using its hostname and the .local suffix. The default hostname on a fresh Raspberry Pi OS install is raspberrypi, so by default any Raspberry Pi running Raspberry Pi OS responds to: `ping raspberrypi.local`. If you change the system hostname of your Raspberry Pi using Raspberry Pi Configuration, raspi-config, or /etc/hostname, Avahi updates the .local mDNS address. If you don’t remember the hostname of your Raspberry Pi, you can install Avahi on another device, then use avahi-browse to browse all the hosts and services on your local network.

Find devices with Network Mapper command (nmap)
    * To install on Linux, install via `apt install nmap`. Mac/Windows should download from [nmap](http://nmap.org/download.html): `sudo nmap -sn 192.168.1.0/24`, Use the `-sn` flag to run a ping scan on the entire subnet range.

Android/iOS App [Fing](https://itunes.apple.com/gb/app/fing-network-scanner/id430921107?mt=8) can see a list with all the devices connected to your network.

Enable the SSH server
    * UI based solution: from the Preferences menu, launch Raspberry Pi Configuration. Navigate to the Interfaces tab. Select Enabled next to SSH.
    * Terminal based solution: enter `sudo raspi-config` in a terminal window. Select Interfacing Options, navigate to and select SSH. Choose Yes. Select Ok. Choose Finish.
    * Manually, Create an empty file named ssh in the boot partition: `sudo touch /boot/firmware/ssh`, `sudo reboot`

Remote access with [Raspberry Pi Connect](https://www.raspberrypi.com/documentation/services/connect.html): 
```bash
sudo apt update
sudo apt upgrade
sudo apt install rpi-connect
sudo reboot
rpi-connect signin
```
Visit `connect.raspberrypi.com`, sign in to Connect using your Raspberry Pi ID.

Install SMB
```bash
sudo apt install samba samba-common-bin smbclient cifs-utils
lkk@raspberrypi:~ $ chmod 0740 Developer/
lkk@raspberrypi:~ $ sudo smbpasswd -a lkk
sudo nano /etc/samba/smb.conf
sudo service samba restart #restart samba if needed
```
At the end of the file, add the following to share the folder, giving the remote user read/write permissions. Replace the <username> placeholder with the username of your primary user account:
```bash
[share]
    path = /home/<username>/Developer
    read only = no
    public = yes
    writable = yes
```
Connect to the SMB folder via "smb://192.168.86.174"

Install the following packages (default is already installed)
```bash
$ sudo apt install python3-smbus
$ sudo apt-get install -y i2c-tools
$ sudo apt-get install python3-dev python3-rpi.gpio
```

Create python virtual environment and jupyterlab
```bash
lkk@raspberrypi:~ $ mkdir mypyvenv
lkk@raspberrypi:~ $ python3 -m venv ./mypyvenv/
lkk@raspberrypi:~ $ source ./mypyvenv/bin/activate
(mypyvenv) lkk@raspberrypi:~ $ pip install RPi.GPIO
pip3 install gpiozero
pip install lgpio
(mypyvenv) lkk@raspberrypi:~ $ pip install jupyterlab
(mypyvenv) lkk@raspberrypi:~ $ jupyter kernelspec list
(mypyvenv) lkk@raspberrypi:~ $ pip install ipykernel
(mypyvenv) lkk@raspberrypi:~ $ ipython kernel install --user --name=mypyvenv
(mypyvenv) lkk@raspberrypi:~ $ jupyter lab --ip='192.168.86.174' --port=8080 --no-browser
```

Setup the Repo:
```bash
lkk@raspberrypi:~ $ mkdir Developer
lkk@raspberrypi:~ $ cd Developer && git clone https://github.com/lkk688/edgeAI.git
```

Visual Studio Code on Raspberry Pi: [link](https://code.visualstudio.com/docs/setup/raspberry-pi). You can adjust the zoom level in VS Code with the View > Appearance > Zoom commands. Install language extensions (e.g., Python) and Jupyter Extension in VSCode.
```bash
sudo apt update
sudo apt install code
code . #open vscode for one folder
```

Enable VSCode remote tunnel to the Raspberry Pi: 1) Click the User icon in the VSCode in Raspberry Pi, turn on "Remote Tunnel Access", login via Github account; 2) Install "Remote Development" extension in the host VSCode. After the extension is installed, you can see the remote tunnel in the host VSCode.

Enable tunnel as a service (https://code.visualstudio.com/docs/remote/tunnels#_how-can-i-ensure-i-keep-my-tunnel-running)
```bash
code tunnel service install
code tunnel service uninstall
```


Serial
```bash
pip install pyserial
(mypyvenv) lkk@raspberrypi:~/Developer/edgeAI $ python -m serial.tools.miniterm

--- Available ports:
---  1: /dev/ttyUSB0         'USB Serial'

pip install ipywidgets
```
Numpy error of "Original error was: libopenblas.so.0: cannot open shared object file"
```bash
sudo apt install libatlas3-base
sudo apt-get install libopenblas-dev
pip3 install numpy
```


Install OpenCV from https://pypi.org/project/opencv-python/#history
```bash
pip install opencv-python==4.7.0.72
import cv2
print(cv2.version)
4.7.0
``` 

```bash
rpicam-hello -t 0
```

Raspberry Pi picamera2 python: https://github.com/raspberrypi/picamera2/tree/main
