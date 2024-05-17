# edgeAI

# CPU and Memory

Jupyter notebook sample code (also available on Google Colab)
* [C GCC Compiler and Assenbly.ipynb](/cpumemory/C_GCC_Compiler_and_Assenbly.ipynb), [colab link](https://colab.research.google.com/drive/1S7dEt_c4RXU-iKunZmAEYb9A_sZcyq7e?usp=sharing)
* [C Review and Memory Layout.ipynb](/cpumemory/C_Review_and_Memory_Layout.ipynb), [colab link](https://colab.research.google.com/drive/1NkU7XPSIwvwlpsXU3p8zBcdBP4XZgNGk?usp=sharing)
* [CPU System.ipynb](/cpumemory/CPU_System.ipynb), [colab link](https://colab.research.google.com/drive/178iJ4B-Qj8NcRiriNPObpQW3DGDOPSlj?usp=sharing)
* [Memory Mountain.ipynb](/cpumemory/memorymountainv2.ipynb), [colab link](https://colab.research.google.com/drive/14F7NXa3bzeYqK0cIkikLtlkmEDncP-je?usp=sharing)
* [matrixmultiple.ipynb](/cpumemory/matrixmultiple.ipynb), [colab link](https://colab.research.google.com/drive/1qQhOGBLSOZJfCGRjo0lNQYMvQMvgujTx?usp=sharing)
    * different matrix multiple versions, BLAS, Intel OneAPI MKL, Intel OneAPI DPC++, Intel OneAPI basekit
* [Multiprocess_and_Concurrent.ipynb](/cpumemory/Multiprocess_and_Concurrent.ipynb), [colab link](https://colab.research.google.com/drive/1gA3KjRGxGlFCQtLcZ1IUCvCH1eF9U9g0?usp=sharing)

# GPU
Jupyter notebook sample code (also available on Google Colab)
* [GPU_and_Cuda_C++.ipynb](/cpumemory/GPU_and_Cuda_C++.ipynb), [colab link](https://colab.research.google.com/drive/1yzRjf8_9TIH4TFO48ooMyvLGHZdvN6cs?usp=sharing)

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
