## Camera Setup
Check JetPack version
```bash
$ dpkg-query --show nvidia-l4t-core
nvidia-l4t-core	36.4.3-20250107174145
```
## Linux kernel camera stack (V4L2)
To enable and use two cameras connected to the CSI (Camera Serial Interface) on the NVIDIA Jetson Orin Nano, you’ll need to configure and access them properly through the Jetson’s Linux kernel camera stack (V4L2) or GStreamer. The camera modules must be supported via the device tree (DTB). For many modules (e.g., Raspberry Pi Cam v2), support is built-in. For third-party cameras, vendors often provide DTB overlays or instructions.
```bash
sudo apt update
sudo apt install v4l-utils
v4l2-ctl --list-formats-ext
v4l2-ctl --list-devices
```