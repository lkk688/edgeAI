# SO-ARM101 Bring-Up and Teleoperation Tutorial (Jetson Orin Nano)

Last updated: 2026-08-02

A step-by-step guide to going from an unplugged SO-ARM101 to keyboard jogging,
PS5 DualSense control, and leader-arm teleoperation on the Jetson Orin Nano.

Do the steps in order. Each one is a checkpoint: if it fails, fix it before
moving on, because every later step assumes the earlier ones passed.

Companion files:

```text
jetson/robotics/so101_unified_teleop.py   teleop modes + interactive menu
jetson/robotics/lerobot_datapipeline.py   dataset QA + remote training
jetson/robotics/jetson_devices.py         camera + Bluetooth controller tool
jetson/robotics/JETSON_ORIN_NANO_SETUP.md host/environment reference
jetson/robotics/SO101_TRAINING.md         which policies run on an SO-101
jetson/robotics/policy_bench/             scripts behind the measured numbers
```

Both scripts are also copied to the Jetson home directory:

```text
/home/cmpe/so101_unified_teleop.py
/home/cmpe/jetson_devices.py
```

## What You Need

- SO-ARM101 follower arm with its controller board and power supply
- USB cable from the follower board to the Jetson
- Optional: a second SO-ARM101 arm to use as a leader
- Optional: a PS5 DualSense controller
- Clear space around the arm. It will move.

## Shortcut: The Interactive Menu

Everything below is also reachable from one guided menu, which shows the
current port/state at the top and walks the steps in order:

```bash
ssh jetsonorin
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/so101_unified_teleop.py interactive
```

```text
  1. detect serial ports
  2. set follower port
  3. lerobot-find-port  (unplug/replug to identify an arm)
  4. lerobot-info       (environment check)
  5. calibrate follower (lerobot-calibrate)
  6. keyboard jogging test
  7. PS5 controller status / pairing
  8. PS5 local teleop   (dry run, no robot)
  9. PS5 local teleop   (real robot)
  10. leader-arm teleop (lerobot-teleoperate)
```

Options 3, 4, 5 and 10 shell out to the real LeRobot CLIs, so the menu is a
launcher rather than a reimplementation. The rest of this tutorial explains
what each step does and what "good" looks like.

## Step 0: Environment

```bash
ssh jetsonorin
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
lerobot-info
```

Use `~/lerobot-py310-cuda` for hardware work on the Orin Nano. It has CUDA
torch, TensorRT, Feetech servo support, `pyrealsense2`, `pygame` and `evdev`.
`~/lerobot-py312` has newer LeRobot but CPU-only torch and no RealSense.

Checkpoint: `lerobot-info` prints a LeRobot version without tracebacks.

## Step 1: Connect the Arm and Find Its Port

Before plugging anything in, list what is already there:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

On a clean Jetson this prints nothing. Now connect the follower board's USB
cable and power, then look again:

```bash
ls /dev/ttyACM*
```

You should see a new node, typically `/dev/ttyACM0`.

If two arms are connected and you cannot tell which is which, let LeRobot
identify them by asking you to unplug one:

```bash
lerobot-find-port
```
The port of this MotorsBus is '/dev/ttyACM0'

Checkpoint: you know the follower's port. Export it so the rest of the
tutorial can be copy-pasted:

```bash
export FOLLOWER=/dev/ttyACM0
```

If no node appears:

- check the arm's power supply, not just USB
- `dmesg | tail -20` right after plugging in shows whether the kernel saw it
- try a different USB cable; some are charge-only

## Step 1b: Serial Port Permission

`/dev/ttyACM*` is owned by `root:dialout` with mode 0660. Unlike `/dev/input`,
systemd-logind does **not** attach a uaccess ACL to serial ports, so being
logged in locally does not help. Check:

```bash
python ~/so101_unified_teleop.py check-motors --follower-port $FOLLOWER
```

If it reports a permission error, add yourself to `dialout` once:

```bash
sudo usermod -aG dialout $USER
```

Then log out and back in, or for the current shell only:

```bash
newgrp dialout
```

Checkpoint: `python3 -c "import serial; serial.Serial('$FOLLOWER', 1000000).close(); print('OK')"`
prints `OK`.

## Step 2: Set Up the Motors (First Time Only)

A brand-new arm needs each Feetech servo assigned its ID:

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER
```

Follow the prompts: it asks you to connect one motor at a time.

### Already configured on another computer? Skip to Step 3

**Motor IDs are stored in each servo's non-volatile EEPROM, not on the host.**
They travel with the hardware, so an arm configured on a different computer is
still configured here. There is nothing to re-do and no cable to unplug.

The SO-101 convention, from `lerobot.robots.so_follower`, numbers the arm from
the base up:

```text
id 1  shoulder_pan     (base, bottom)
id 2  shoulder_lift
id 3  elbow_flex
id 4  wrist_flex
id 5  wrist_roll
id 6  gripper          (top)
```

So reading top to bottom you should see 6, 5, 4, 3, 2, 1. Verify without
writing anything:

```bash
python ~/so101_unified_teleop.py check-motors --follower-port $FOLLOWER
```

This scans every supported baud rate and reports which IDs answer:

```text
baud 1000000: ids [1, 2, 3, 4, 5, 6]
    id 1  shoulder_pan   ok
    id 2  shoulder_lift  ok
    ...
All six SO-101 servos responded. The arm is already configured -
skip `lerobot-setup-motors` and go straight to calibration.
```

Only run `lerobot-setup-motors` if IDs are missing, duplicated, or wrong.

What does *not* travel with the arm is **calibration** — that is a JSON file on
the host, keyed by `--robot.id`. So you still need Step 3 on this machine.

## Step 3: Calibrate

Calibration records each joint's range so positions mean the same thing every
session. Unlike motor IDs, this lives **on the host**, so it must be done on
each computer you drive the arm from. Do it once per arm per machine, and again
if you rebuild the arm.

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER \
  --robot.id=so101_follower
```

Calibration saved to /home/cmpe/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json

Run this in a real terminal: it is interactive and needs you at the arm.

**Safety first:** the very first thing calibration does is `disable_torque()`.
The arm goes completely limp and will fall under gravity. Support it with your
hand before you start, or lower it to a resting pose first.

What it asks, in order:

1. If a calibration already exists for this id, it offers to reuse it. Press
   ENTER to reuse, or type `c` then ENTER to redo it.
2. `Move <robot> to the middle of its range of motion and press ENTER.`
   Put every joint roughly mid-travel. This sets the half-turn homing offsets.
3. `Move all joints except 'wrist_roll' sequentially through their entire
   ranges of motion. Recording positions. Press ENTER to stop...`
   Sweep each joint slowly to both mechanical limits and back. Press ENTER
   when done.

Note that **`wrist_roll` is deliberately skipped** — it is treated as a
full-turn joint and hardcoded to the full 0..4095 range, so you do not need to
move it.

**The two prompts want opposite things. Do not confuse them:**

| prompt | what to do | why |
|---|---|---|
| ① "move to the middle … press ENTER" | park mid-travel, **do NOT sweep** | sweeping here inflates the multi-turn counter and `set_half_turn_homings()` overflows |
| ② "move all joints … press ENTER to stop" | **sweep every joint fully** | this is the step that records the ranges; homing has re-centred everything on 2047, so sweeping is safe |

Pressing ENTER straight away at prompt ② "succeeds" but silently records a
useless calibration — spans of 1–5 counts instead of ~2000:

```text
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   2045 |   2046 |   2047      <- span 2, wrong
shoulder_lift   |   2047 |   2051 |   2052
```

Watch the live table grow while you sweep. A good result reaches roughly
2047 ± 1000 on each joint:

```text
shoulder_pan    |   1025 |   2034 |   2854      <- span 1829, correct
shoulder_lift   |    866 |    866 |   3197      <- span 2331
```

### Verifying a calibration

Compare the saved spans against an arm that already works. The mechanics are
identical between leader and follower, so the spans should be in the same
ballpark; only the absolute range shifts.

```bash
python3 - <<'PY'
import json
L = json.load(open("/home/cmpe/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader.json"))
F = json.load(open("/home/cmpe/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"))
for k in F:
    ls = L[k]["range_max"] - L[k]["range_min"]
    fs = F[k]["range_max"] - F[k]["range_min"]
    bad = L[k]["range_min"] < 0 or L[k]["range_max"] > 4095 or (k != "wrist_roll" and ls < 500)
    print("%-14s leader span %5d | follower span %5d | %s" % (k, ls, fs, "CHECK" if bad else "ok"))
PY
```

Three things must hold: every span comparable to the reference, every range
inside `0..4095`, and every `homing_offset` within `±2047`.

The result is written to a directory named after the robot *class* (`so_follower`),
with the filename taken from `--robot.id`:

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot-id>.json
```
Calibration saved to /home/cmpe/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json

Later commands only need `--robot.id=so101_follower` to find it.

Checkpoint: the command exits cleanly and that JSON file exists.

## Step 4: Keyboard Jogging (First Motion Test)

Do this before any controller work. It moves one joint a fixed step per
keypress, which is the most predictable way to confirm the arm responds.

### You do not need a keyboard plugged into the Jetson

Run it straight from your normal SSH session. This mode reads keys with
`termios` / `tty.setcbreak` on stdin, so the keystrokes are the ones you type in
your own terminal, forwarded over SSH. No X11, no `DISPLAY`, no `pynput`, and
nothing attached to the Jetson.

That is precisely why this mode exists. LeRobot's own `keyboard` teleoperator
imports `pynput`, which grabs keys from a graphical session, and it refuses to
load headless:

```text
No DISPLAY set. Skipping pynput import.
ImportError: pynput blocked intentionally due to no display.
```

So use `so101_unified_teleop.py keyboard` over SSH; do not use
`lerobot-teleoperate --teleop.type=keyboard` over SSH.

Two practical notes:

- It needs a real interactive terminal. Piping into it (`echo x | ...`) or
  running it from a non-tty context fails fast with
  `RuntimeError: keyboard mode needs an interactive terminal`.
- **Use `tmux` or `screen`.** Each keypress is a discrete step, so network
  latency is harmless, but if the SSH connection drops mid-session the process
  is killed by SIGHUP and the `finally: robot.disconnect()` cleanup never runs,
  leaving the arm torqued and holding position. A multiplexer keeps the session
  alive across a dropped link:

  ```bash
  ssh jetsonorin
  tmux new -s arm
  # if you get disconnected: ssh back in, then `tmux attach -t arm`
  ```

```bash
python ~/so101_unified_teleop.py keyboard \
  --follower-port $FOLLOWER \
  --robot-id so101_follower \
  --joint-step-deg 2.0
```

Keys:

```text
q / a    shoulder_pan   +/-
w / s    shoulder_lift  +/-
e / d    elbow_flex     +/-
r / f    wrist_flex     +/-
t / g    wrist_roll     +/-
y / h    gripper        +/-
space    print the current goal
x, ESC   exit
```

Start with a single `q` press and watch the arm. One press should produce a
small, controlled 2-degree move.

Safety notes for this step:

- `--max-relative-target` defaults to 8.0, which caps how far a single command
  can jump. Leave it low while you are learning the arm.
- Keep a hand near the power switch for the first few presses.
- If a joint runs into a mechanical limit, stop and re-run calibration.

Checkpoint: every joint moves in both directions and the gripper opens/closes.

## Step 5: PS5 DualSense Control on the Jetson

This drives the arm from a DualSense paired directly to the Jetson. No Mac and
no network hop is involved, unlike `mac-ps5-client`.

### 5a. Pair the Controller

Use the device tool's PS5 control panel:

```bash
python ~/jetson_devices.py bluetooth ps5
```

It opens an interactive panel showing current status and offering pair,
connect, disconnect, unpair, live input test, and the button mapping. The same
actions are available as one-shot flags for scripting:

```bash
python ~/jetson_devices.py bluetooth ps5 --status
python ~/jetson_devices.py bluetooth ps5 --pair
python ~/jetson_devices.py bluetooth ps5 --connect
python ~/jetson_devices.py bluetooth ps5 --disconnect
python ~/jetson_devices.py bluetooth ps5 --unpair
python ~/jetson_devices.py bluetooth ps5 --test
```

To pair: turn the controller off, then hold **PS + Create** until the light bar
flashes blue in double pulses, then choose pair in the menu.

Two behaviours worth knowing, both handled automatically by the tool:

- Right after bonding, the controller connects back *inbound* and BlueZ asks
  the agent to authorize the HID service. If that prompt goes unanswered for
  ~30 seconds the link is dropped with `Access denied`.
- A bonded controller that has gone to sleep will not answer an outbound
  connect and fails with `br-connection-create-socket`. Press the PS button to
  wake it; it then connects on its own.

Checkpoint: `--status` reports `connected paired trusted` and shows an input
node such as `/dev/input/event7`.

### 5b. Check the Buttons

```bash
python ~/jetson_devices.py bluetooth ps5 --test
```

This prints the full mapping table, then a live panel:

```text
  Left stick   LX -0.84 ..#########|
               LY -0.92 .##########|
  Right stick  RX +0.80            |#########..
               RY +0.00 ...........|
  Triggers     L2  0.00 ......................
               R2  0.78 #################.....
  D-pad        < . . .   LEFT

  Held         Square  R1
  Last press   Square           [BTN_SOUTH]
```

On a plain SSH session without a pty the panel cannot redraw in place, so it
prints events instead. Use `ssh -t jetsonorin` for the live panel.

Checkpoint: pressing Square shows `Square`, squeezing R2 fills the R2 bar.

### 5c. Dry Run First

Never go straight to the arm. This reads the controller and prints the goals it
*would* send, with no robot connected:

```bash
python ~/so101_unified_teleop.py ps5-local --follower-port $FOLLOWER --dry-run
```

Hold **L1** and move the left stick. The status line should switch from `HOLD`
to `MOVING` and `shoulder_pan` should climb:

```text
MOVING shoulder_pan=  14.95 shoulder_lift=   0.00 ... held=[l1]
```

Checkpoint: `MOVING` appears only while L1 is held, and the joint you expect is
the one that changes.

### 5d. Drive the Arm

```bash
python ~/so101_unified_teleop.py ps5-local \
  --follower-port $FOLLOWER \
  --robot-id so101_follower \
  --joint-speed-deg-s 15
```

Start at a low `--joint-speed-deg-s` (the default is 30 deg/s) until you trust
the mapping.

Control layout:

```text
Left stick   left/right   shoulder_pan
Left stick   up/down      shoulder_lift
Right stick  up/down      elbow_flex
Right stick  left/right   wrist_flex
D-pad        left/right   wrist_roll
R2 trigger                open gripper
L2 trigger                close gripper

L1 (hold)                 DEADMAN - the arm only moves while held
Circle                    E-STOP  - freeze until Options is pressed
Options                   clear E-STOP
PS                        quit
```

The deadman is the important one: release L1 and the arm stops accepting input
immediately. Useful options:

```text
--deadman-button r1        use a different hold button
--deadzone 0.12            widen the stick deadzone if the sticks drift
--joint-speed-deg-s 15     slower joints
--gripper-speed-s 20       slower gripper
--limits-json '{"shoulder_pan.pos": [-90, 90]}'   clamp a joint
--dry-run                  print goals instead of moving
```

## Step 6: Leader-Arm Teleoperation

With a second SO-ARM101 as leader, LeRobot's own teleoperate CLI is the best
path. Calibrate the leader first, with its own id.

**The leader is a *teleoperator*, not a robot.** In LeRobot the two are separate
registries, and `--robot.type` only accepts followers. Use `--teleop.*`:

```bash
python ~/so101_unified_teleop.py center-check --port /dev/ttyACM1


lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so101_leader


NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1025 |   2034 |   2854
shoulder_lift   |    866 |    866 |   3197
elbow_flex      |    879 |   3066 |   3082
wrist_flex      |   1811 |   2673 |   3169
gripper         |   2036 |   2047 |   3344
Calibration saved to /home/cmpe/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader.json
```

Passing `--robot.type=so101_leader` fails with:

```text
lerobot-calibrate: error: argument --robot.type: invalid choice: 'so101_leader'
  (choose from 'openarm_follower', ..., 'so101_follower', ...)
```

The two registries are visible in `lerobot-calibrate --help`:

```text
--robot.type   {..., so100_follower, so101_follower, ...}   followers
--teleop.type  {..., so100_leader,   so101_leader,   ...}   leaders
```

Leader calibration is stored separately from follower calibration:

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json
```

The leader uses the same servo IDs 1..6 as the follower, so the read-only check
works on it too:

```bash
python ~/so101_unified_teleop.py check-motors \
  --follower-port /dev/ttyACM1 --role leader --robot-id so101_leader
```

Calibrating the leader is the same physical procedure as Step 3: it goes limp,
you centre it, then sweep every joint except `wrist_roll` through its range.

Then:

```bash
python ~/so101_unified_teleop.py leader \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1 \
  --robot-id so101_follower \
  --leader-id so101_leader
```

This delegates to `lerobot-teleoperate`, which exists in both LeRobot 0.4.4 and
0.5.x.

## Step 7: Record a Dataset

With leader teleop working, you can record LeRobot-format datasets. The wrapper
picks the cameras up automatically so you never have to look up device paths.

### 7a. Wrist camera only

The USB UVC camera mounted on the follower arm:

```bash
python ~/so101_unified_teleop.py record \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1 \
  --repo-id local/so101_pick \
  --task "Pick up the cube and place it in the box" \
  --cameras wrist \
  --episodes 5 --episode-time-s 20 --reset-time-s 5
```

Add `--dry-run` to print the `lerobot-record` command without running it.

**`leader` mode records nothing.** It is teleoperation only — if you see
`Teleop loop time: 16.75ms (60 Hz)` scrolling, you are in `leader`, not
recording. Recording is the `record` mode above, and it announces every phase.

### Knowing what is happening

`lerobot-record` prints its phase changes (`Recording episode N`,
`Reset the environment`) among a lot of other logging, which is easy to miss.
The wrapper watches for those and renders one always-visible status line:

```text
  REC  episode 2/5  RECORDING  [##############..........]  12.3/20s
       episode 2/5  RESET      [##############..........]   3.1/5s
```

`REC` on the left means it is actually capturing. The bar counts down the
current phase, so you always know how long is left and what comes next. Every
line the wrapper does not recognise is still printed, so warnings and errors are
never hidden — only the repetitive loop-time and config-dump spam is filtered.

Pass `--status false` for raw `lerobot-record` output.

### Controlling it: the timer is the real control

LeRobot binds right arrow (next episode), left arrow (re-record) and escape
(stop) through **pynput**, which reads the **X server**, not your terminal.
Over SSH those keypresses go to your terminal emulator and never reach it, so
they usually do nothing.

Worse, the detection is misleading: `is_headless()` only checks whether
`pynput` can be *imported*. With X11 forwarding it imports fine and returns
`False`, so LeRobot creates a listener that then dies on the forwarded display:

```text
Exception in thread Thread-5:
  File ".../pynput/_util/xorg.py", line 403, in _run
    self._context = dm.record_create_context(
AttributeError: record_create_context
```

That traceback is harmless — the X11 RECORD extension is not available over
forwarding, so the listener thread simply exits. Recording continues normally;
you just have no keyboard shortcuts.

So plan around the timer instead:

- size `--episode-time-s` to how long the task actually takes, plus a margin
- size `--reset-time-s` to how long you need to put the objects back
- `Ctrl-C` always works if you need to abort
- to drop a bad episode, just let it finish and delete it afterwards, or
  re-record extra episodes and ignore the bad ones

If you want working shortcuts, run from a terminal on a monitor physically
attached to the Jetson, where pynput and your keyboard share the same X server.

### 7b. Adding the RealSense as a scene view

Plug in the D435i and add it to `--cameras`. The serial is looked up through the
SDK, so nothing to configure:

```bash
python ~/so101_unified_teleop.py record \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1 \
  --repo-id local/so101_pick \
  --task "Pick up the cube and place it in the box" \
  --cameras wrist,scene \
  --episodes 5 --episode-time-s 20
```

Both streams are recorded as separate video keys in the dataset
(`observation.images.wrist` and `observation.images.scene`).

The RealSense stream is stored as **`scene`**, not `desk`, because pretrained
checkpoints declare exact observation keys — FLUX 3 Action SO-101 wants
`observation.images.scene` + `observation.images.wrist` — and a key that does
not match is silently dropped. `--cameras wrist,desk` still works as an alias
and still writes `scene`. See `SO101_TRAINING.md`.

Presets: `wrist` is the USB UVC camera on the arm (OpenCV backend), `scene` is
the RealSense (its own SDK), `none` records joint states only. Override the
auto-detection with `--wrist-device /dev/videoN` or `--scene-serial <serial>`,
and rotate a badly-mounted camera with `--wrist-rotation 90|180|270`.

### Choosing the frame rate

**30 fps works.** `--fps` defaults to 30, and `--camera-fps` to 30.

Camera rate and dataset rate are still separate settings — the camera runs at a
mode it advertises, the record loop can run slower — but they do not need to be
different here.

The reason this needs saying: benchmarking the camera with LeRobot's blocking
`read()` suggests 30 Hz is impossible, and that is misleading. `read()` clears
an event and waits for the *next* frame, so a 29.5 fps camera cannot satisfy a
30 Hz caller. But recording never calls it — `get_observation()` uses the
non-blocking `read_latest()`:

```python
for cam_key, cam in self.cameras.items():
    obs_dict[cam_key] = cam.read_latest()
```

Measured both ways on the same camera:

```text
read()        @ 30Hz:  141/148 late (95%), worst overrun 82.9 ms
read()        @ 25Hz:    1/125 late ( 1%)
read_latest() @ 30Hz:    0/150 late ( 0%), 146/150 unique frames
read_latest() @ 25Hz:    0/125 late ( 0%), 125/125 unique frames
```

At 30 Hz about 3% of cycles reuse the previous frame, because the camera
delivers ~29.5 fps against a 30 Hz request. That is a negligible cost and well
worth it: **30 fps is what most SO-101 community datasets use**, so recording at
30 keeps the option of mixing them open.

Note the camera only accepts modes it advertises — `--camera-fps 25` fails with
`failed to set fps=25 (actual_fps=30.0)`. Change `--fps`, not `--camera-fps`.

`MJPG` is essential. Without it the camera negotiates YUYV and collapses to
about 5 fps at high resolution. The wrapper sets it by default.

### Is the USB camera the bottleneck?

Not really. The Innomaker U20CAM is rated 30 fps and delivers ~29.5. Measured
three ways at 640x480 MJPG:

```text
v4l2-ctl --stream-mmap=4   27.5 - 28.3 fps   (only 4 buffers)
GStreamer v4l2src          29.74 fps
OpenCV CAP_V4L2 default    29.5 fps
OpenCV with BUFFERSIZE=1   14.8 fps          (starves itself)
```

The low numbers are artefacts of buffer depth, not the camera. Do **not** set
`CAP_PROP_BUFFERSIZE=1` hoping to reduce latency — here it halves throughput.

What the camera genuinely lacks is *headroom*: every mode it advertises tops out
at 30 fps, so there is no 60 fps option to comfortably guarantee 30. If you want
that margin, pick a camera offering 60 fps at 640x480 — the RealSense D435i
colour stream does, which is one reason to use it as the fixed desk view.

### Multiple cameras (dual-arm setups)

Two things matter, and neither is raw bandwidth. At 640x480 MJPG this camera
produces about 18 KB per frame, roughly 4.4 Mbit/s — trivial.

**1. Bus assignment is decided by the device, not by which port you use.**

The Orin Nano dev kit advertises 4x USB 3.2 Type-A ports, yet `lsusb -t` shows
everything crowded onto a 480M bus while the 10000M bus looks empty:

```text
Bus 02  10000M  root_hub  ->  4-Port USB 3.0 Hub          (0bda:0489)
Bus 01    480M  root_hub  ->  4-Port USB 2.0 Hub          (0bda:5489)
                                |__ USB Single Serial      12M   bcdUSB 1.10
                                |__ Innomaker U20CAM      480M   bcdUSB 2.00
                                |__ USB Single Serial      12M   bcdUSB 1.10
                              |__ Bluetooth Radio          12M   bcdUSB 1.00
```

That is not a fault. A USB 3.x Type-A connector carries **two independent sets of
wires**: the legacy D+/D- pair and the SuperSpeed pairs. The board's hub
therefore enumerates **twice** — `0bda:5489` is its USB 2.0 face on Bus 01 and
`0bda:0489` is its USB 3.0 face on Bus 02. Same physical chip, same four
sockets.

A device lands on whichever bus matches **its own** capability, which
`bcdUSB` reports:

| device | bcdUSB | lands on |
|---|---|---|
| Innomaker UVC camera | 2.00 | Bus 01 (480M) |
| CH340 serial (both arms) | 1.10 | Bus 01 (12M) |
| Bluetooth | 1.00 | Bus 01 (12M) |
| RealSense D435i | 3.x | **Bus 02 (5000M)** |

So **moving a USB 2.0 camera to a different port changes nothing** — it has no
SuperSpeed wires to use. Earlier advice to "put the cameras on the USB3 bus" was
wrong: the only way onto Bus 02 is to use a device that is itself USB 3. The
D435i proves the routing works — it enumerated on Bus 002 at 5000M.

**2. On USB 2.0, what limits you is isochronous reservation, not data rate.**

A UVC camera reserves bandwidth by its negotiated `wMaxPacketSize`, regardless of
how much data it actually sends. This camera offers seven alternate settings:

```text
alt   bytes/microframe   reserves      max cameras on one USB2 bus
 1              128        8.2 Mbps         46
 2              256       16.4 Mbps         23
 3              800       51.2 Mbps          7
 4             1600      102.4 Mbps          3
 5             2400      153.6 Mbps          2
 6             3072      196.6 Mbps          1
```

(USB 2.0 high speed = 8000 microframes/s; periodic transfers may claim at most
80% of 480 Mbps, so the budget is 384 Mbps.)

The driver picks the alt setting from the requested resolution and frame rate.
At 640x480 MJPG the actual payload is only ~4.3 Mbit/s, but a high alt setting
may still reserve 150–200 Mbit/s — which is why a second camera can fail with
`No space left on device` while `lsusb` shows almost no traffic.

Practical consequences for a dual-arm rig:

- Request the **lowest resolution you can live with**; it lowers the alt setting
  and therefore the reservation.
- Two 640x480 MJPG cameras on one USB2 bus is usually fine. Two 1080p cameras
  usually is not.
- If you need more, use **USB 3.0 cameras** so they land on Bus 02 and stop
  competing for the USB2 periodic budget altogether.

**2. USB UVC cameras have no hardware sync.** Each free-runs on its own clock, so
frames drift relative to each other. LeRobot handles this by giving every camera
a background capture thread and having the record loop take the most recent
frame from each (`read_latest()`), then storing them against one frame index.
Alignment is therefore only as good as the control period — about 33 ms at
30 Hz.

If a dual-arm task needs tighter alignment than that, UVC will not give it to
you. Use cameras with genuine multi-camera sync instead: RealSense D400 series
supports hardware sync over its sync connector, and Orbbec Gemini models offer
multi-camera sync too. For most manipulation learning, 33 ms software alignment
is fine — this only becomes a real problem for fast, bimanual, contact-rich
motions.

### The arm freezes between episodes

Expected, and worth understanding. After each episode LeRobot encodes the video
before starting the next one, and the robot is not being commanded during that
window — so the follower stops following, then jumps to catch up when teleop
resumes. In a real run the gap was about 4–6 s per episode.

The cure is to encode *during* recording instead of after:

```bash
--streaming-encoding true --encoder-threads 2
```

LeRobot suggests this itself in its own log line, and measured here it removes
the gap completely:

```text
without streaming: 2 x 10s episodes took 41 s wall  (11 s of encoding pauses)
with streaming   : 2 x (5s + 2s reset) took 14 s wall  (0 s overhead)
```

A clean run looks like this — phase banners, no pauses, correct count:

```text
[00:34:12] ===== episode 1/2  RECORDING (5s) =====
[00:34:17] ===== episode 1/2  RESET (2s) =====
[00:34:19] ===== episode 2/2  RECORDING (5s) =====
[00:34:25] ===== episode 2/2  SAVING =====

Episodes recorded: 2  re-records: 0
```

Do **not** reach for `--vcodec libx264` expecting it to help. Benchmarked on
this Orin Nano with realistic camera-like content, 200 frames (10 s at 20 fps):

```text
libsvtav1   1.3 s   18 KB
libx264     2.7 s  103 KB
```

`libsvtav1` is both faster and far smaller here despite the board having no
hardware encoder, so it stays the default. (Measuring with random noise instead
reverses the result — noise is a pathological case for x264 and not
representative of camera footage.)

### Where it goes

`--push-to-hub` defaults to **false**, so nothing is uploaded and no
`hf auth login` is needed. Any `a/b` repo id works locally. Use `--root` to
choose the directory, and `--resume` to append episodes to an existing dataset.

## Step 8: Review What You Recorded

The Jetson is headless, so serve the viewer over the network and open it from
your laptop:

```bash
python ~/so101_unified_teleop.py view \
  --repo-id local/so101_pick \
  --episode 0
```

It prints the exact URL to open, for example:

```text
  http://192.168.5.206:9090/?url=rerun%2Bhttp%3A%2F%2F192.168.5.206%3A9876%2Fproxy
```

**The `?url=` part is essential.** rerun serves two things: a web viewer on
`--web-port` and the actual data stream on `--grpc-port`. Opening the bare page
leaves the viewer pointing at *the browser's own* localhost, so you get rerun's
built-in example data instead of your recording — which looks like the command
worked but showed the wrong dataset. LeRobot's own log is no help here; it
prints a literal placeholder:

```text
Connect to a Rerun Server: rerun rerun+http://IP:9876/proxy
```

with `IP` never substituted. The wrapper resolves the Jetson's real address and
builds the full URL for you.

Use `--mode local` if you are sitting at a monitor plugged into the Jetson, and
`--web-port` / `--grpc-port` if those are taken. The viewer holds the ports until
you Ctrl-C it — a leftover instance causes
`message proxy server crashed: Address already in use`, so kill it before
starting another.

### Recording into an existing dataset

`LeRobotDataset.create()` calls `mkdir(exist_ok=False)`, so pointing `--repo-id`
at a dataset that already exists aborts with a `FileExistsError` buried under two
stacked tracebacks. The wrapper checks first and tells you the options:

```text
ERROR: dataset already exists at
  /home/cmpe/.cache/huggingface/lerobot/local/so101_test

Pick one:
  --resume                      append episodes to it
  --repo-id local/<new_name>    record a separate dataset
  rm -rf <path>                 discard it and start over
```

To replay recorded actions on the real arm (it will move):

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower \
  --dataset.repo_id=local/so101_pick \
  --dataset.episode=0
```

## Step 9: LeLab - the Web UI

Everything above is CLI. [LeLab](https://github.com/huggingface/leLab) is
HuggingFace's official web UI for LeRobot, Apache 2.0, and its docs state it is
**compatible only with SO-ARM101** — exactly this arm. It removes the two
biggest CLI annoyances: recording control and previewing.

### Install and run

```bash
# on the Jetson
~/.local/bin/uv tool install git+https://github.com/huggingface/leLab.git
export PATH=$HOME/.local/bin:$PATH
lelab --no-open          # stop with: lelab --stop
```

It binds to **127.0.0.1:8000 only**, so from another machine use an SSH tunnel
rather than exposing it:

```bash
# on your laptop
ssh -N -L 18000:127.0.0.1:8000 jetsonorin
# then open http://localhost:18000
```

### Why it solves the recording problems

The controls are **HTTP endpoints, not keyboard shortcuts**, so the pynput/X11
problem from Step 7 simply does not exist:

```text
/recording-exit-early           end this episode now, move to the next
/recording-rerecord-episode     discard and redo the current episode
/recording-status               is it recording, which episode
/camera-feed/{cam_key}          live camera preview
/datasets  /dataset-info  /delete-dataset
/start-inference  /inference-status
/jobs/training  /jobs/{id}/logs  /jobs/{id}/metrics-history  /jobs/{id}/checkpoints
```

62 endpoints on a FastAPI backend. It also picks up datasets recorded by the
CLI, so the two workflows share the same storage.

### The gap

`/jobs/runners/hardware` returns `{"authenticated": false, "flavors": []}` —
training is **local or HF Jobs (cloud) only**. It cannot submit to your own GPU
box. That is what `lerobot_datapipeline.py` below is for.

## Step 10: Dataset QA and Remote Training

```text
jetson/robotics/lerobot_datapipeline.py
```

Robot-agnostic — it reads joint names, action dimensions and camera keys from
the dataset metadata, so the same pipeline serves the SO-ARM101, a Gaoqing arm,
a SeeedStudio re-Arm, or anything else recording in LeRobot format.

Runs on the Jetson, submits training to a remote GPU machine. The transport is
behind a `RemoteBackend` abstract class (`check / push / pull / exec / submit`)
with an `SSHRsyncBackend` implementation, so swapping rsync for Slurm, S3 or a
cloud API later means writing one class, not touching the pipeline.

```bash
python ~/lerobot_datapipeline.py interactive     # guided menu over everything

python ~/lerobot_datapipeline.py inspect                     # all local datasets
python ~/lerobot_datapipeline.py check   --repo-id local/so101_pick
python ~/lerobot_datapipeline.py compat  --repo-id local/so101_pick --with <hub-id> ...
python ~/lerobot_datapipeline.py merge   --repo-id A --with B --new-repo-id local/merged
python ~/lerobot_datapipeline.py remotes --check
python ~/lerobot_datapipeline.py push    --repo-id local/so101_pick
python ~/lerobot_datapipeline.py train   --repo-id local/so101_pick --policy smolvla
python ~/lerobot_datapipeline.py status  --run <run-id> --follow
python ~/lerobot_datapipeline.py fetch   --run <run-id>
```

Remotes live in `~/.lerobot_pipeline.json`; `cmpe28803` is just the first entry.
Submitted runs are tracked in `~/.lerobot_runs.json`. Training is launched under
`nohup` with its pid recorded, so it survives losing the SSH connection.

### What the quality check catches

`check` reads `meta/info.json`, `meta/episodes/*.parquet` and `meta/stats.json`
and reports things that quietly ruin training. A real example from this repo's
own first recording:

```text
  joint                 min      max    range      std
  shoulder_pan.pos    10.15    13.76     3.60     1.32  barely used (3% of median)
  shoulder_lift.pos  -102.20    51.12   153.32    39.48
  elbow_flex.pos     -57.80    95.96   153.76    41.20

  [WARN] barely exercised: shoulder_pan.pos
  [INFO] the demos never really used these joints, so a policy will
  [INFO] not learn to move them - vary the object/target placement
```

The base rotation moved 3.6 degrees across five episodes while the other joints
swept 120–150. No amount of extra episodes fixes that; the object has to be
placed in varied positions.

It also flags missing/empty video files, episodes far shorter than the median
(usually aborted demos), and datasets with no camera at all.

## Mixing Community Datasets

There are many SO-101 datasets on the Hub, and mixing them with your own is a
real way to improve generalisation — [SmolVLA itself was trained on LeRobot
community data](https://huggingface.co/blog/smolvla), and MolmoAct2 curated
1,222 public LeRobot datasets from 377 users.

**But there is a trap that fails silently.** `MultiLeRobotDataset` keeps only
the *intersection* of features and disables the rest:

```python
intersection_features = set(self._datasets[0].features)
for ds in self._datasets:
    intersection_features.intersection_update(ds.features)
...
extra_keys = set(ds.features).difference(intersection_features)
if extra_keys:
    logging.warning(f"keys {extra_keys} of {repo_id} were disabled ...")
```

A camera key not shared by *every* dataset is dropped. Since camera naming is
pure convention (`wrist`, `front`, `laptop`, `top`...), mixing usually leaves
**no** common image key, and the policy trains on proprioception alone while
only emitting a `logging.warning`.

`compat` checks this before you download anything — it pulls only
`meta/info.json` from the Hub:

```bash
python ~/lerobot_datapipeline.py compat --repo-id local/so101_pick \
    --with SGPatil/so101_pick_drop kinam0252/so101_three_strawberries
```

```text
  repo_id                                    src  fps   eps  cameras / action dim
  local/so101_pick                         local   20     5  wrist  | [6]
  SGPatil/so101_pick_drop                    hub   30    46  front  | [6]
  kinam0252/so101_three_strawberries         hub   30    19  front, wrist  | [6]

  [FAIL] no camera key is shared by all datasets
  [INFO] every image stream would be disabled and the policy would train blind
  [WARN] fps differs across datasets: [20, 30]
  [ OK ] action dim matches: [6]
```

So before mixing, line these up:

1. **Camera keys must match exactly.** Rename yours to the community
   convention, or drop the odd stream with
   `lerobot-edit-dataset --operation.type=remove_feature`.
2. **fps should match.** 20 vs 30 changes what one timestep means, which
   matters for action chunking.
3. **Action dim must match.** All SO-101 datasets are `[6]`, so this is usually
   fine — it is the one thing that is naturally compatible.
4. **Record at 30 fps if you plan to mix**, since community datasets mostly use
   30. Our camera tops out near 27 fps (Step 7), so that argues for a faster
   USB camera if community co-training is the goal.

Expect mixed results: extra data helps generalisation, but
[fine-tuning a pretrained VLA on a small real dataset can also hurt](https://medium.com/correll-lab/when-fine-tuning-hurts-failure-modes-of-visuomotor-imitation-learning-on-a-low-cost-robot-5f5013df7c95).
Keep an ACT baseline on your own data alone as the control.

## Alternative: PS5 Controller on a Mac

If the controller is already paired to a Mac, you can keep it there and send
commands over the network. Run the server on the Jetson:

```bash
python ~/so101_unified_teleop.py remote-server \
  --follower-port $FOLLOWER \
  --bind-host 0.0.0.0 \
  --require-deadman true
```

and the client on the Mac:

```bash
python mylerobot/scripts/so101_unified_teleop.py mac-ps5-client \
  --jetson-host jetsonorin --deadman-button 5 --print-events
```

Use `--print-events` first: pygame's axis/button numbering on macOS differs
from the Jetson's evdev numbering, so you may need to adjust `--axis-lx` and
friends.

Prefer `ps5-local` when the controller can be paired to the Jetson directly. It
has no network latency and no chance of a stale-command timeout mid-motion.

## Safety Checklist

Before any powered run:

- clear the workspace within the arm's reach
- keep `--max-relative-target` small (default 8.0) so one bad command cannot
  slam a joint
- know your stop: release L1 (PS5), press Circle for E-STOP, or Ctrl-C
- test in `--dry-run` after changing any mapping or speed
- `--disable-torque-on-disconnect` defaults to true, so the arm goes limp on a
  clean exit; support the arm if it is holding a pose

## Troubleshooting

**No `/dev/ttyACM*` after plugging in the arm.**
Check arm power separately from USB, watch `dmesg | tail -20` while plugging
in, and try another cable.

**`lerobot-calibrate` cannot reach the motors.**
Confirm the port with `lerobot-find-port` and that no other process holds it
(a previous teleop session still running will hold the serial port).

**The controller pairs but no `/dev/input/event*` appears.**
Give it a few seconds, then re-check with `bluetooth ps5 --status`. If the node
exists but is unreadable, see the permissions note below.

**`br-connection-create-socket` when connecting the DualSense.**
The controller is asleep. Press the PS button and let it connect inbound.

**Repeating `WARNING: Relative goal position magnitude had to be clamped to be
safe` during `ps5-local`.**
This meant the joint had hit a mechanical or calibration limit while you kept
pushing the stick. `ps5-local` integrates a goal, `send_action()` returns the
*clamped goal* rather than the *measured* position, so the goal kept marching
past what the joint could reach and parked permanently on the
`max_relative_target` boundary — one warning per control cycle:

```text
'original goal_pos': -112.578    <- where the goal had wound up to
'safe goal_pos'    : -112.527    <- present (-104.527) + the 8 deg cap
```

Fixed by re-reading the measured position each cycle and holding the goal
inside a band around it (`--goal-lead-deg`, default `0.8 x
--max-relative-target`). The arm was never in danger — the clamp was doing its
job — but the goal is now kept honest, which also stops the arm lurching when
you release and re-press the deadman. Update the script if you still see it.

**`ValueError: Magnitude 2993 exceeds 2047 (max for sign_bit_index=11)` during
calibration.**

Calibration's first step, `set_half_turn_homings()`, writes
`Homing_Offset = present_position - 2047` into an **11-bit sign-magnitude**
field, so the magnitude must be ≤ 2047. On Feetech STS3215, `Present_Position`
is itself sign-magnitude with sign bit 15, so a joint parked right on the
0/4095 encoder wrap can read as a small *negative* number — and
`-946 - 2047 = -2993` overflows the field.

In other words: **a joint was sitting at the very edge of its encoder range
when you pressed ENTER**, not in the middle. Check before calibrating:

```bash
python ~/so101_unified_teleop.py center-check --port /dev/ttyACM1
```

This is a read-only live readout. It resolves the true encoder value (it adds
back any `Homing_Offset` already written) and flags anything near the wrap:

```text
  shoulder_pan   raw=    29  offset_needed= -2018  NEAR WRAP
  shoulder_lift  raw=  3792  offset_needed= +1745  OK
  wrist_roll     raw=   110  offset_needed= -1937  NEAR WRAP

  NOT READY - move the flagged joints toward mid-travel
```

Move the flagged joints by hand toward the middle until every row reads `OK`,
then re-run `lerobot-calibrate`. A partially-completed calibration leaves stale
offsets on the motors, but that is harmless: the next run calls
`reset_calibration()` and starts clean.

### Do I need to unbolt the motor?

Usually no. Getting a joint away from the wrap by hand is enough *if* its whole
travel stays inside one turn. What actually breaks calibration is the encoder
crossing 0/4095 **during the range sweep**, because `record_ranges_of_motion()`
then records a discontinuous min/max and the joint behaves erratically
afterwards.

`center-check` detects that for you. Sweep each joint slowly through its full
physical travel while it runs and watch the `seen` column:

```text
  shoulder_pan   raw=  1064  offset=  -983  seen=1064..1064  OK
```

- `seen` grows smoothly across the sweep and no row says `WRAPPED`
  → the joint never crosses the wrap. **No disassembly needed.** Park it at
  mid-travel and calibrate.
- A row flips to `WRAPPED` (the reading jumped by more than 2000 counts, which
  no real joint can do between samples)
  → that joint's travel straddles the wrap. Unbolt its horn, rotate it roughly
  half a turn so mid-travel lands near 2047, and refit.

Chasing a merely "OK" number without sweeping is not enough: a joint parked at
1064 still wraps if it can travel more than about 1000 counts further in the
decreasing direction. Sweep first, then decide.

### The multi-turn counter is the real cause - power cycle clears it

STS3215 servos keep a **multi-turn counter in RAM**. Once a joint is driven past
0 or 4095 the reading keeps counting rather than wrapping:

```text
shoulder_lift  3791 .. 5079    <- 5079 is past the 12-bit encoder maximum
wrist_flex     -693 ..  513    <- and this one went below zero
```

`set_half_turn_homings()` then computes `5079 - 2047 = 3032`, which cannot fit
in the 11-bit `Homing_Offset` field, and calibration dies. Nothing in LeRobot
resets that counter — `reset_calibration()` only zeroes `Homing_Offset`, a
different register.

**Unplug the arm's power and USB, wait ~5 seconds, plug back in.** The counter
is volatile, so the servos come back reporting true single-turn positions:

```text
shoulder_lift  5079  ->  3791     back inside 0..4095
```

The trap is that *checking* the arm by sweeping every joint is what accumulates
the counter in the first place. So:

1. Power cycle the arm.
2. Run `center-check` and park each joint near mid-travel — **do not sweep**.
3. Run `lerobot-calibrate` and press ENTER straight away.
4. Sweep the joints only when calibration asks you to. By then the homing
   offsets have re-centred every joint on 2047, so sweeping is safe.

If a joint is still out of range immediately after a power cycle, that one
genuinely needs its horn re-indexed.

### Finding which horns are mis-indexed

Use the follower (or any arm that calibrates cleanly) as the reference. The
*span* of each joint is set by the mechanics and should match between arms; only
the *centre* differs when a horn is fitted at the wrong servo angle.

A real leader arm, measured with `sync_read` while each joint was swept:

```text
joint             leader seen    span   centre   action
shoulder_pan     2065..2066         1     2065   ok (already re-seated)
shoulder_lift    3793..6208      2415     5000   rotate horn +1143 (+100 deg)
elbow_flex       -896..1304      2200      204   rotate horn +1843 (+162 deg)
wrist_flex        481..1002       521      741   ok
wrist_roll       3056..3059         3     3057   ok
gripper          3492..3494         2     3493   ok

follower (calibrates fine):
  shoulder_pan   904..3047  span 2143
  shoulder_lift  753..3149  span 2396
  elbow_flex    1059..3093  span 2034
```

The spans line up (2415 vs 2396, 2200 vs 2034) — the mechanics are identical.
Only `shoulder_lift` and `elbow_flex` sit in the wrong part of the encoder, so
only those two horns need re-seating.

Rotation is modulo one turn, so `center-check` reports the *smallest* equivalent
angle: `+100 deg` rather than `-260 deg`.

Two-phase workflow, because surveying and calibrating want opposite things:

1. **Survey** — power cycle, then sweep every joint fully while `center-check`
   runs. Sweeping is what reveals a bad range, and it deliberately racks up the
   multi-turn counter. Note which joints overflow and by how much.
2. **Fix** — re-seat the flagged horns.
3. **Calibrate** — power cycle again to clear the counter, park every joint at
   mid-travel, do *not* sweep, then run `lerobot-calibrate` and press ENTER
   straight away.

### Stale homing offsets make readings lie

`set_half_turn_homings()` writes offsets **one motor at a time**, so a run that
fails part-way leaves real offsets on the motors it already reached and zeros on
the rest. `Present_Position` is then reported relative to those offsets, and
reconstructing the raw encoder from `Present + Homing_Offset` is unreliable — the
servo clamps its own output, so the arithmetic can produce impossible values
like `-1416` on a device whose encoder only spans `0..4095`.

Symptom: readings that swing wildly between runs, or joints that look broken one
minute and fine the next, with no hardware change in between.

`center-check` now calls `reset_calibration()` at startup — the same first step
`lerobot-calibrate` performs — so `Present_Position` *is* the raw encoder and the
numbers mean what they say. If you are ever debugging this by hand, zero the
offsets before trusting any position you read:

```python
bus.reset_calibration()          # Homing_Offset = 0 on every motor
raw = bus.read("Present_Position", motor, normalize=False)
```

This is safe and non-destructive: calibration rewrites the offsets anyway.

### `wrist_roll` is a special case - never unbolt it for this

`set_half_turn_homings()` runs on **every** motor, but `record_ranges_of_motion()`
skips `wrist_roll`; `calibrate()` hardcodes its range instead:

```python
full_turn_motor = "wrist_roll"
unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]
range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
range_mins[full_turn_motor] = 0
range_maxes[full_turn_motor] = 4095
```

So how far `wrist_roll` was swept is irrelevant — only where it sits **when you
press ENTER** matters. It spins freely, so if it reads near the wrap just
**turn it by hand** toward 2047. No disassembly, ever. `center-check` knows this
and says so instead of telling you to re-seat the horn.

### Diagnosing a real case

A leader arm that failed twice, showing how each joint differs:

```text
shoulder_pan   seen=-1057..1081   travel straddles the origin
                                  -> horn half a turn out, must be re-seated
wrist_roll     raw=90             free-spinning, excluded from range recording
                                  -> just rotate it by hand toward 2047
gripper        seen= 3811..4218   407-count travel spilling past 4095
                                  -> re-index its horn, or verify it stays
                                     inside one turn across a full open/close
```

The follower arm calibrated first is a useful reference for what "good" looks
like — every `range_min`/`range_max` comfortably inside `0..4095`:

```text
shoulder_pan   904..3047     elbow_flex   1059..3093
shoulder_lift  753..3149     wrist_flex   2047..3218
```

**Repeating `Relative goal position magnitude had to be clamped to be safe`
during leader-arm teleop.**

Harmless, and expected with a tight cap. A follower always trails the leader by
several degrees under load, so any cap smaller than that lag clamps almost
continuously while you move. Reading one warning back:

```text
'wrist_roll': {'original goal_pos': 40.396, 'safe goal_pos': 41.011}
```

`safe = present - 8`, so the follower was at 49.011 deg while the leader asked
for 40.396 — a lag of 8.6 deg against an 8 deg cap.

Note that **LeRobot's own default is no cap at all** (`max_relative_target:
float | dict | None = None` in `SOFollowerRobotConfig`). This script imposes a
cap because `keyboard` and `ps5-local` integrate a goal, where a runaway must
not be able to slam a joint. Leader teleop does not have that failure mode — the
leader *is* the reference — so it defaults to a roomier 30 deg:

```text
leader                        max_relative_target = 30.0
keyboard / ps5-local          max_relative_target = 8.0
leader --max-relative-target none   -> no cap (LeRobot stock behaviour)
```

Raise or disable it if the warnings persist; keep some cap if you want
protection against a bad leader reading.

**`ERROR: Controller disconnected: [Errno 9] Bad file descriptor` when quitting
with PS.**
Harmless, and now silenced. The reader thread sits blocked in evdev's
`read_loop()`; shutdown closes the device out from under it, so the blocked read
raised `EBADF` and the handler reported it as a disconnect. The controller was
fine — that was just the shutdown racing itself.

**Input node exists but Python cannot read it.**
On this Jetson it can, because systemd-logind grants a `user:cmpe:rw-` uaccess
ACL to input devices for the active local seat session. If you ever run fully
headless with no console login, that ACL is not applied and you need:

```bash
sudo usermod -aG input cmpe   # then log out and back in
```

**Button labels look wrong.**
This kernel has no `hid_playstation` driver, so the DualSense enumerates as a
generic HID gamepad and the kernel's `BTN_*` names do not line up with the
printed PlayStation labels. Both scripts translate using the DualSense HID
report order, so they show `Square`, `Cross`, `Circle`, `Triangle` correctly.
`bluetooth ps5 --test` prints the full table if you need the raw codes.

**`gamepad-local` does not work with the follower.**
LeRobot's built-in `gamepad` teleoperator emits end-effector deltas
(`delta_x`, `delta_y`, `delta_z`, `gripper`) and expects a robot with inverse
kinematics. A plain `so101_follower` consumes joint positions
(`shoulder_pan.pos` and friends), so the action spaces do not match. Use
`ps5-local`, which is joint-space, instead.

## What Has Been Verified

Verified on `cmpe-jetson` on 2026-08-02:

```text
PS5 pairing, trust, connect, live input        DualSense A0:FA:9C:8B:2B:6F
Button label mapping                           Circle/Triangle/L1/R1/L2/R2 confirmed
Axis mapping                                   L2=ABS3, R2=ABS4 confirmed by absinfo
DualSenseReader autodetect + live streaming     /dev/input/event7
ps5-local control loop (deadman, E-STOP,
  speed scaling, gripper clamp)                 verified with a simulated pad
```

Not yet run against a powered SO-ARM101, because no arm was connected at the
time of writing:

```text
Step 2 lerobot-setup-motors
Step 3 lerobot-calibrate
Step 4 keyboard jogging against real motors
Step 5d ps5-local against real motors
Step 6 leader-arm teleoperation
```

Those steps use the same code paths that were unit-tested and dry-run, but
treat the first powered run as a real test: start slow, keep the workspace
clear, and stay on the stop.
