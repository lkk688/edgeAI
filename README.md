# edgeAI

# Raspberry Pi Setup
Configure Raspberry Pi to enable SSH, I2C, SPI and others: [RaspiConfig](https://www.raspberrypi.com/documentation/computers/configuration.html#the-raspi-config-tool)

Interactive pinout diagram: [pinout](https://pinout.xyz)

Check raspberry pi's IP address via "hostname -I"

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

