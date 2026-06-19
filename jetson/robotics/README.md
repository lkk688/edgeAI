# Robotics on Jetson — SO‑ARM101 + LeRobot

Code and hardware notes for physical robotics labs on the Jetson Orin Nano.

| File | What it is |
|------|------------|
| [`so101_unified_teleop.py`](so101_unified_teleop.py) | Unified SO‑ARM101 teleop helper (leader / keyboard / remote‑server / mac‑ps5‑client / gamepad‑local). Works with LeRobot 0.4.4 and 0.5.x. |
| [`JETSON_ORIN_NANO_SETUP.md`](JETSON_ORIN_NANO_SETUP.md) | Hardware/setup log for the lab box (`cmpe-jetson`): LeRobot envs, CUDA/TensorRT, RealSense, CSI cameras, encoding. |

📖 **Full tutorial:** [LeRobot & SO‑ARM101 on Jetson Orin Nano](../../docs/curriculum/05d_lerobot_so101.md)

Quick start (on the Jetson):
```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=$HOME/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python so101_unified_teleop.py --help
```
