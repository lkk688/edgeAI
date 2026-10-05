#!/usr/bin/env python3
"""Unified SO-ARM101 teleoperation helper.

This script is intentionally conservative about LeRobot internals:

- `leader` delegates to the stable `lerobot-teleoperate` CLI, which is present
  in both LeRobot 0.4.4 and 0.5.x.
- `keyboard` and `remote-server` use the common SO follower API that is shared
  by LeRobot 0.4.4 and 0.5.x:
      lerobot.robots.so_follower.SOFollower
      lerobot.robots.so_follower.SOFollowerRobotConfig

Typical Jetson Orin Nano environment:

    source ~/lerobot-py310-cuda/bin/activate
    export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH

Leader-arm teleop:

    python mylerobot/scripts/so101_unified_teleop.py leader \
      --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
      --robot-id so101_follower --leader-id so101_leader

Terminal keyboard joint jogging:

    python mylerobot/scripts/so101_unified_teleop.py keyboard \
      --follower-port /dev/ttyACM0 --robot-id so101_follower

Remote server on Jetson:

    python mylerobot/scripts/so101_unified_teleop.py remote-server \
      --follower-port /dev/ttyACM0 --bind-host 0.0.0.0

Mac PS5 client:

    python mylerobot/scripts/so101_unified_teleop.py mac-ps5-client \
      --jetson-host jetsonorin.local

HTTP example:

    python mylerobot/scripts/so101_unified_teleop.py api-post \
      --url http://jetsonorin:8765/command \
      --json '{"delta": {"shoulder_pan.pos": 2.0}, "deadman": true}'
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import queue
import re
import select
import signal
import shutil
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.parse
import urllib.request
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOG = logging.getLogger("so101_unified_teleop")

MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_max_relative_target(value: str) -> float | None:
    """Accept a number, or 'none'/'off' to disable the safety clamp.

    LeRobot's own default for so101_follower is None (no clamp). This script
    imposes a small cap by default because `keyboard` and `ps5-local` integrate
    a goal and a runaway there should not be able to slam a joint. Leader-arm
    teleop is different: the leader *is* the reference, and the follower always
    lags it by a few degrees, so a tight cap just produces a stream of
    "Relative goal position magnitude had to be clamped" warnings.
    """
    if value.strip().lower() in {"none", "off", "disable", "disabled"}:
        return None
    return float(value)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def load_json_arg(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def clamp(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


def lerobot_executable(name: str) -> list[str]:
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    module = name.replace("-", "_")
    return [sys.executable, "-m", f"lerobot.scripts.{module}"]


def add_if_value(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = str(value).lower()
    cmd.append(f"{key}={value}")


def run_lerobot_teleoperate(args: argparse.Namespace, teleop_type: str) -> int:
    cmd = lerobot_executable("lerobot-teleoperate")
    cmd += [
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.use_degrees={str(args.use_degrees).lower()}",
        f"--robot.disable_torque_on_disconnect={str(args.disable_torque_on_disconnect).lower()}",
        f"--fps={args.fps}",
    ]
    add_if_value(cmd, "--robot.calibration_dir", args.calibration_dir)
    add_if_value(cmd, "--robot.max_relative_target", args.max_relative_target)
    add_if_value(cmd, "--teleop_time_s", args.time_s)
    add_if_value(cmd, "--display_data", args.display_data)

    cmd.append(f"--teleop.type={teleop_type}")
    if teleop_type == "so101_leader":
        cmd += [
            f"--teleop.port={args.leader_port}",
            f"--teleop.id={args.leader_id}",
            f"--teleop.use_degrees={str(args.use_degrees).lower()}",
        ]
        add_if_value(cmd, "--teleop.calibration_dir", args.leader_calibration_dir)

    cmd.extend(args.extra_arg or [])
    LOG.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def import_so101_runtime():
    """Import the LeRobot SO follower classes lazily.

    The import path is shared by LeRobot 0.4.4 and 0.5.x.
    """

    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    return SOFollower, SOFollowerRobotConfig


def make_so101_robot(args: argparse.Namespace):
    SOFollower, SOFollowerRobotConfig = import_so101_runtime()
    kwargs = {
        "port": args.follower_port,
        "id": args.robot_id,
        "calibration_dir": Path(args.calibration_dir) if args.calibration_dir else None,
        "disable_torque_on_disconnect": args.disable_torque_on_disconnect,
        "max_relative_target": args.max_relative_target,
        "use_degrees": args.use_degrees,
        "cameras": {},
    }
    allowed = {f.name for f in fields(SOFollowerRobotConfig)}
    kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return SOFollower(SOFollowerRobotConfig(**kwargs))


def observation_to_goal(obs: dict[str, Any]) -> dict[str, float]:
    goal: dict[str, float] = {}
    for key in MOTOR_KEYS:
        if key in obs:
            goal[key] = float(obs[key])
    missing = [key for key in MOTOR_KEYS if key not in goal]
    if missing:
        raise RuntimeError(f"robot observation is missing motor keys: {missing}")
    return goal


def default_limits() -> dict[str, tuple[float | None, float | None]]:
    return {
        "shoulder_pan.pos": (None, None),
        "shoulder_lift.pos": (None, None),
        "elbow_flex.pos": (None, None),
        "wrist_flex.pos": (None, None),
        "wrist_roll.pos": (None, None),
        "gripper.pos": (0.0, 100.0),
    }


def parse_limits(value: str | None) -> dict[str, tuple[float | None, float | None]]:
    limits = default_limits()
    raw = load_json_arg(value, {})
    for key, pair in raw.items():
        if key not in MOTOR_KEYS:
            raise ValueError(f"unknown motor key in limits: {key}")
        if pair is None:
            limits[key] = (None, None)
        else:
            limits[key] = (pair[0], pair[1])
    return limits


def apply_limits(goal: dict[str, float], limits: dict[str, tuple[float | None, float | None]]) -> None:
    for key, (lo, hi) in limits.items():
        if key in goal:
            goal[key] = clamp(float(goal[key]), lo, hi)


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self, timeout_s: float) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            return "esc"
        return ch


KEYBOARD_DELTAS = {
    "q": ("shoulder_pan.pos", 1.0),
    "a": ("shoulder_pan.pos", -1.0),
    "w": ("shoulder_lift.pos", 1.0),
    "s": ("shoulder_lift.pos", -1.0),
    "e": ("elbow_flex.pos", 1.0),
    "d": ("elbow_flex.pos", -1.0),
    "r": ("wrist_flex.pos", 1.0),
    "f": ("wrist_flex.pos", -1.0),
    "t": ("wrist_roll.pos", 1.0),
    "g": ("wrist_roll.pos", -1.0),
    "y": ("gripper.pos", 1.0),
    "h": ("gripper.pos", -1.0),
}


def run_keyboard(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard mode needs an interactive terminal")
    limits = parse_limits(args.limits_json)
    robot = make_so101_robot(args)
    robot.connect(calibrate=not args.no_calibrate)
    try:
        goal = observation_to_goal(robot.get_observation())
        print("Keyboard joint jogging. Press x or ESC to exit.")
        print("  q/a shoulder_pan, w/s shoulder_lift, e/d elbow_flex")
        print("  r/f wrist_flex, t/g wrist_roll, y/h gripper")
        print("  space prints current goal; keys move one step per press.")
        with RawTerminal() as terminal:
            while True:
                key = terminal.read_key(1.0 / max(args.fps, 1))
                if key in {"x", "esc"}:
                    print("\nExiting.")
                    return 0
                if key == " ":
                    print("\n" + json.dumps(goal, indent=2))
                    continue
                if key not in KEYBOARD_DELTAS:
                    continue
                motor, sign = KEYBOARD_DELTAS[key]
                step = args.gripper_step if motor == "gripper.pos" else args.joint_step_deg
                goal[motor] += sign * step
                apply_limits(goal, limits)
                sent = robot.send_action(goal)
                goal.update({k: float(v) for k, v in sent.items() if k in goal})
                print(f"\r{motor}: {goal[motor]:8.2f}", end="", flush=True)
    finally:
        robot.disconnect()


class CommandBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: dict[str, Any] = {}
        self.latest_source = "none"
        self.latest_time = 0.0
        self.delta_queue: queue.Queue[dict[str, float]] = queue.Queue()
        self.estop = False

    def update(self, payload: dict[str, Any], source: str) -> None:
        now = time.monotonic()
        with self.lock:
            if payload.get("estop"):
                self.estop = True
            if payload.get("resume"):
                self.estop = False
            if "delta" in payload:
                self.delta_queue.put({k: float(v) for k, v in payload["delta"].items()})
            self.latest = payload
            self.latest_source = source
            self.latest_time = now

    def snapshot(self) -> tuple[dict[str, Any], str, float, bool, list[dict[str, float]]]:
        deltas: list[dict[str, float]] = []
        while True:
            try:
                deltas.append(self.delta_queue.get_nowait())
            except queue.Empty:
                break
        with self.lock:
            age = time.monotonic() - self.latest_time if self.latest_time else math.inf
            return dict(self.latest), self.latest_source, age, self.estop, deltas


def start_udp_listener(command_buffer: CommandBuffer, host: str, port: int) -> threading.Thread:
    def run() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        LOG.info("UDP command listener on %s:%d", host, port)
        while True:
            data, addr = sock.recvfrom(65535)
            try:
                payload = json.loads(data.decode("utf-8"))
                command_buffer.update(payload, f"udp:{addr[0]}:{addr[1]}")
            except Exception as exc:
                LOG.warning("Ignoring malformed UDP packet from %s: %s", addr, exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def make_http_handler(command_buffer: CommandBuffer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.debug("http: " + fmt, *args)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            latest, source, age, estop, deltas = command_buffer.snapshot()
            self._send_json(
                200,
                {
                    "ok": True,
                    "source": source,
                    "age_s": age,
                    "estop": estop,
                    "latest": latest,
                    "queued_deltas": len(deltas),
                },
            )

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body.decode("utf-8"))
                if self.path == "/estop":
                    payload["estop"] = True
                elif self.path == "/resume":
                    payload["resume"] = True
                elif self.path not in {"/", "/command", "/estop", "/resume"}:
                    self._send_json(404, {"ok": False, "error": "unknown path"})
                    return
                command_buffer.update(payload, f"http:{self.client_address[0]}")
                self._send_json(200, {"ok": True})
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})

    return Handler


def start_http_server(command_buffer: CommandBuffer, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_http_handler(command_buffer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    LOG.info("HTTP command server on http://%s:%d", host, port)
    return server


def axes_to_joint_delta(
    payload: dict[str, Any],
    dt_s: float,
    joint_speed_deg_s: float,
    wrist_roll_speed_deg_s: float,
    gripper_speed_s: float,
    deadzone: float,
) -> dict[str, float]:
    axes = payload.get("axes") or {}
    joint_axes = payload.get("joint_axes") or {}
    delta = {key: 0.0 for key in MOTOR_KEYS}

    for key, value in joint_axes.items():
        if key in delta:
            delta[key] += apply_deadzone(float(value), deadzone) * joint_speed_deg_s * dt_s

    lx = apply_deadzone(float(axes.get("lx", 0.0)), deadzone)
    ly = apply_deadzone(float(axes.get("ly", 0.0)), deadzone)
    rx = apply_deadzone(float(axes.get("rx", 0.0)), deadzone)
    ry = apply_deadzone(float(axes.get("ry", 0.0)), deadzone)
    wrist = apply_deadzone(float(axes.get("wrist_roll", 0.0)), deadzone)
    l2 = float(axes.get("l2", 0.0))
    r2 = float(axes.get("r2", 0.0))

    delta["shoulder_pan.pos"] += lx * joint_speed_deg_s * dt_s
    delta["shoulder_lift.pos"] += -ly * joint_speed_deg_s * dt_s
    delta["elbow_flex.pos"] += -ry * joint_speed_deg_s * dt_s
    delta["wrist_flex.pos"] += rx * joint_speed_deg_s * dt_s
    delta["wrist_roll.pos"] += wrist * wrist_roll_speed_deg_s * dt_s
    delta["gripper.pos"] += (r2 - l2) * gripper_speed_s * dt_s
    return delta


def run_remote_server(args: argparse.Namespace) -> int:
    limits = parse_limits(args.limits_json)
    command_buffer = CommandBuffer()
    start_udp_listener(command_buffer, args.bind_host, args.udp_port)
    http_server = start_http_server(command_buffer, args.bind_host, args.http_port)

    robot = None if args.dry_run else make_so101_robot(args)
    if robot is not None:
        robot.connect(calibrate=not args.no_calibrate)
        goal = observation_to_goal(robot.get_observation())
    else:
        goal = {key: 0.0 for key in MOTOR_KEYS}

    last_loop = time.monotonic()
    last_print = 0.0
    try:
        while True:
            now = time.monotonic()
            dt_s = max(now - last_loop, 1.0 / max(args.fps, 1))
            last_loop = now

            payload, source, age, estop, queued_deltas = command_buffer.snapshot()
            stale = age > args.command_timeout_s
            allowed = not estop and not stale
            if args.require_deadman and not payload.get("deadman", False):
                allowed = False

            if allowed:
                if "action" in payload:
                    for key, value in payload["action"].items():
                        if key in goal:
                            goal[key] = float(value)
                for one_delta in queued_deltas:
                    for key, value in one_delta.items():
                        if key in goal:
                            goal[key] += float(value)
                axis_delta = axes_to_joint_delta(
                    payload=payload,
                    dt_s=dt_s,
                    joint_speed_deg_s=args.joint_speed_deg_s,
                    wrist_roll_speed_deg_s=args.wrist_roll_speed_deg_s,
                    gripper_speed_s=args.gripper_speed_s,
                    deadzone=args.deadzone,
                )
                for key, value in axis_delta.items():
                    goal[key] += value
                apply_limits(goal, limits)
                if robot is not None:
                    sent = robot.send_action(goal)
                    goal.update({k: float(v) for k, v in sent.items() if k in goal})

            if now - last_print > args.status_period_s:
                state = "ESTOP" if estop else "STALE" if stale else "DEADMAN" if not allowed else "RUN"
                print(f"{state:7s} source={source:24s} age={age:5.2f}s goal={goal}")
                last_print = now

            time.sleep(max(1.0 / max(args.fps, 1) - (time.monotonic() - now), 0.0))
    except KeyboardInterrupt:
        return 0
    finally:
        http_server.shutdown()
        if robot is not None:
            robot.disconnect()


def trigger_to_unit(value: float) -> float:
    """Normalize common pygame trigger ranges to 0..1."""
    value = float(value)
    if value < -0.05:
        return (value + 1.0) / 2.0
    return clamp(value, 0.0, 1.0)


def run_mac_ps5_client(args: argparse.Namespace) -> int:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Install pygame on the Mac first: python -m pip install pygame") from exc

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found. Pair/connect the PS5 controller to the Mac first.")

    joystick = pygame.joystick.Joystick(args.joystick_index)
    joystick.init()
    print(f"Using joystick: {joystick.get_name()}")
    print(f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.jetson_host, args.udp_port)
    period = 1.0 / max(args.fps, 1)

    try:
        while True:
            start = time.monotonic()
            pygame.event.pump()
            axes = {
                "lx": joystick.get_axis(args.axis_lx) * args.scale_lx,
                "ly": joystick.get_axis(args.axis_ly) * args.scale_ly,
                "rx": joystick.get_axis(args.axis_rx) * args.scale_rx,
                "ry": joystick.get_axis(args.axis_ry) * args.scale_ry,
                "l2": trigger_to_unit(joystick.get_axis(args.axis_l2)) if args.axis_l2 >= 0 else 0.0,
                "r2": trigger_to_unit(joystick.get_axis(args.axis_r2)) if args.axis_r2 >= 0 else 0.0,
                "wrist_roll": 0.0,
            }
            if joystick.get_numhats() > 0:
                hat_x, _hat_y = joystick.get_hat(0)
                axes["wrist_roll"] = float(hat_x)
            elif args.wrist_negative_button >= 0 and args.wrist_positive_button >= 0:
                neg = joystick.get_button(args.wrist_negative_button)
                pos = joystick.get_button(args.wrist_positive_button)
                axes["wrist_roll"] = float(pos - neg)

            deadman = True
            if args.deadman_button >= 0:
                deadman = bool(joystick.get_button(args.deadman_button))

            payload = {
                "type": "ps5_axes",
                "deadman": deadman,
                "axes": axes,
                "buttons": {
                    "estop": bool(joystick.get_button(args.estop_button)) if args.estop_button >= 0 else False,
                },
                "t": time.time(),
            }
            if payload["buttons"]["estop"]:
                payload["estop"] = True
            sock.sendto(json.dumps(payload).encode("utf-8"), target)

            if args.print_events:
                button_values = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
                axis_values = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
                hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
                print(f"\raxes={axis_values} buttons={button_values} hats={hats} deadman={deadman}", end="")

            time.sleep(max(period - (time.monotonic() - start), 0.0))
    except KeyboardInterrupt:
        return 0
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# camera discovery + LeRobot dataset recording
# ---------------------------------------------------------------------------

REALSENSE_USB_VENDOR = "8086"


def _v4l_usb_ids(node: str) -> tuple[str | None, str | None]:
    """USB vendor/product for a /sys/class/video4linux/videoN node, if any.

    CSI cameras hang off tegra rather than USB and return (None, None), which is
    what distinguishes them from UVC and RealSense devices.
    """
    path = os.path.realpath(os.path.join(node, "device"))
    for _ in range(6):
        vendor = os.path.join(path, "idVendor")
        if os.path.exists(vendor):
            try:
                with open(vendor) as f:
                    vid = f.read().strip()
                with open(os.path.join(path, "idProduct")) as f:
                    pid = f.read().strip()
                return vid, pid
            except OSError:
                return None, None
        path = os.path.dirname(path)
        if path == "/":
            break
    return None, None


def detect_uvc_camera() -> str | None:
    """First plain USB UVC capture node, skipping CSI and RealSense.

    A UVC device exposes several nodes; the capture one has sysfs `index` 0
    (the others are metadata). RealSense is excluded here because it is driven
    through its own SDK, not OpenCV.
    """
    for node in sorted(glob.glob("/sys/class/video4linux/video*"),
                       key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        vid, _pid = _v4l_usb_ids(node)
        if vid is None or vid == REALSENSE_USB_VENDOR:
            continue
        try:
            with open(os.path.join(node, "index")) as f:
                if f.read().strip() != "0":
                    continue
        except OSError:
            continue
        return "/dev/" + os.path.basename(node)
    return None


def detect_realsense_serial() -> str | None:
    """Serial of the first RealSense the SDK can see."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None
    try:
        for dev in rs.context().query_devices():
            return dev.get_info(rs.camera_info.serial_number)
    except Exception as exc:
        LOG.debug("RealSense query failed: %s", exc)
    return None


def build_cameras(names: list[str], args: argparse.Namespace) -> dict[str, dict]:
    """Turn --cameras names into the dict LeRobot's --robot.cameras expects."""
    cams: dict[str, dict] = {}
    for name in names:
        name = name.strip().lower()
        if not name or name == "none":
            continue
        if name == "wrist":
            path = args.wrist_device or detect_uvc_camera()
            if not path:
                raise RuntimeError(
                    "No USB UVC camera found for 'wrist'. Plug it in, or pass "
                    "--wrist-device /dev/videoN. Check with:\n"
                    "  python ~/jetson_devices.py camera list"
                )
            cams["wrist"] = {
                "type": "opencv",
                "index_or_path": path,
                "width": args.width,
                "height": args.height,
                "fps": args.camera_fps,
                # Without MJPG this camera negotiates YUYV and drops to ~5 fps
                # at high resolution.
                "fourcc": args.wrist_fourcc,
                "rotation": args.wrist_rotation,
            }
        elif name in ("scene", "desk"):
            serial = args.desk_serial or detect_realsense_serial()
            if not serial:
                raise RuntimeError(
                    "No RealSense found for 'scene'. Plug it in, or pass "
                    "--desk-serial <serial>. Needs pyrealsense2 in this env."
                )
            # Always stored as "scene". FLUX 3 Action SO-101 and several other
            # pretrained checkpoints declare exactly observation.images.scene +
            # observation.images.wrist, and a camera key that does not match is
            # silently dropped when datasets are combined, so recording under the
            # convention now keeps those checkpoints usable later. "desk" stays
            # accepted on the command line as the old spelling.
            cams["scene"] = {
                "type": "intelrealsense",
                "serial_number_or_name": serial,
                "width": args.width,
                "height": args.height,
                "fps": args.camera_fps,
            }
        else:
            raise RuntimeError(
                f"unknown camera preset '{name}' (use wrist, scene or none)")
    return cams


def cmd_record(args: argparse.Namespace) -> list[str]:
    cams = build_cameras((args.cameras or "").split(","), args)
    cmd = lerobot_executable("lerobot-record")
    cmd += [
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.use_degrees={str(args.use_degrees).lower()}",
    ]
    add_if_value(cmd, "--robot.max_relative_target", args.max_relative_target)
    if cams:
        cmd.append("--robot.cameras=" + json.dumps(cams))
    cmd += [
        "--teleop.type=so101_leader",
        f"--teleop.port={args.leader_port}",
        f"--teleop.id={args.leader_id}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.fps={args.fps}",
        f"--dataset.push_to_hub={str(args.push_to_hub).lower()}",
        f"--dataset.vcodec={args.vcodec}",
        f"--dataset.streaming_encoding={str(args.streaming_encoding).lower()}",
    ]
    add_if_value(cmd, "--dataset.encoder_threads", args.encoder_threads)
    add_if_value(cmd, "--dataset.root", args.root)
    if args.resume:
        cmd.append("--resume=true")
    if args.display_data:
        cmd.append("--display_data=true")
    cmd.extend(args.extra_arg or [])
    return cmd


class RecordStatus:
    """Live panel over lerobot-record's output.

    lerobot-record announces phase changes through log_say(): "Recording episode
    N", "Reset the environment", "Re-record episode", "Stop recording". Those
    lines scroll past among a lot of other logging, so this watches for them and
    renders one always-visible status line with a countdown, while still letting
    every unrecognised line through so nothing is hidden.
    """

    RE_EPISODE = re.compile(r"Recording episode (\d+)")
    RE_RESET = re.compile(r"Reset the environment")
    RE_REREC = re.compile(r"Re-record episode")
    RE_STOP = re.compile(r"Stop recording")
    # noise that would otherwise scroll the panel away
    RE_NOISE = re.compile(
        r"loop time:"
        r"|^\s*$"
        r"|^\s*'[a-z_]+':"
        r"|^[{}]\s*$"
        r"|^Svt\[/"          # SVT-AV1 banner, printed once per episode
        r"|^Svt\["
        r"|^Map:\s"           # huggingface datasets progress bar
        r"|^\[mp4 @ ")

    def __init__(self, total: int, episode_s: float, reset_s: float):
        self.total = total
        self.episode_s = episode_s
        self.reset_s = reset_s
        self.episode = 0
        self.started = 0
        self.suppress = False
        self.noted = False
        self.pending_tb: str | None = None
        self.phase = "starting"
        self.phase_started = time.monotonic()
        self.rerecorded = 0
        self.done = False
        self.tty = sys.stdout.isatty()
        self._drawn = False

    RE_PYNPUT = re.compile(
        r"pynput|Xlib|DisplayNameError|record_create_context"
        r"|this platform is not supported|X server running"
        r"|Try one of the following|Switching to headless"
        r"|Headless environment detected|is_headless|control_utils")
    RE_TRACEBACK = re.compile(r"^\s*(File \"|Traceback|\w+Error|raise |from \. import|"
                              r"backend = |\*\s|self\.|import )")

    def feed(self, line: str) -> None:
        # A traceback header arrives before the frame that names pynput, so hold
        # it for one line and discard it if the pynput diagnostic follows.
        if line.startswith("Traceback (most recent call last)"):
            self.pending_tb = line
            return
        if self.pending_tb is not None:
            pending, self.pending_tb = self.pending_tb, None
            if not self.RE_PYNPUT.search(line):
                self._passthrough(pending)

        # We unset DISPLAY on purpose, so LeRobot's "cannot import pynput"
        # diagnostic and its traceback are expected. Say it once, plainly.
        if self.RE_PYNPUT.search(line):
            self.suppress = True
            if not self.noted:
                self.noted = True
                self._passthrough(
                    "  [note] keyboard shortcuts unavailable (headless by "
                    "design) - episodes are timed; Ctrl-C to stop")
            return
        if self.suppress:
            if self.RE_TRACEBACK.search(line) or not line.strip():
                return
            self.suppress = False

        m = self.RE_EPISODE.search(line)
        if m:
            self.episode = int(m.group(1))
            self.started += 1
            self._set("RECORDING")
            return
        if self.RE_RESET.search(line):
            self._set("RESET")
            return
        if self.RE_REREC.search(line):
            self.rerecorded += 1
            self._set("RE-RECORD")
            return
        if self.RE_STOP.search(line):
            self._set("SAVING")
            self.done = True
            return
        if self.RE_NOISE.search(line):
            return
        self._passthrough(line)

    def _set(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        budget = self.budget()
        span = f" ({budget:.0f}s)" if budget else ""
        stamp = time.strftime("%H:%M:%S")
        self._passthrough(
            f"[{stamp}] ===== episode {self.episode + 1}/{self.total}  "
            f"{phase}{span} =====")

    def _passthrough(self, line: str) -> None:
        if self.tty and self._drawn:
            sys.stdout.write("\r\033[2K")
            self._drawn = False
        if line:
            print(line)

    def budget(self) -> float | None:
        if self.phase == "RECORDING":
            return self.episode_s
        if self.phase in ("RESET", "RE-RECORD"):
            return self.reset_s
        return None

    def draw(self) -> None:
        if not self.tty or self.done:
            return
        elapsed = time.monotonic() - self.phase_started
        budget = self.budget()
        if budget:
            bar_w = 24
            frac = min(1.0, elapsed / budget)
            bar = "#" * int(frac * bar_w) + "." * (bar_w - int(frac * bar_w))
            timing = f"[{bar}] {elapsed:5.1f}/{budget:.0f}s"
        else:
            timing = f"{elapsed:5.1f}s"
        marker = "REC" if self.phase == "RECORDING" else "   "
        line = (f"  {marker}  episode {self.episode + 1}/{self.total}  "
                f"{self.phase:<10} {timing}")
        width = max(40, shutil.get_terminal_size((100, 24)).columns - 1)
        sys.stdout.write("\r\033[2K" + line[:width])
        sys.stdout.flush()
        self._drawn = True

    def finish(self) -> None:
        if self.tty and self._drawn:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
        print(f"\nEpisodes recorded: {self.started}  "
              f"re-records: {self.rerecorded}")


def run_with_status(cmd: list[str], args: argparse.Namespace,
                    env: dict | None = None) -> int:
    status = RecordStatus(args.episodes, args.episode_time_s, args.reset_time_s)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            status.feed(raw.rstrip())

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        while proc.poll() is None:
            status.draw()
            time.sleep(0.2)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
    thread.join(timeout=2)
    status.finish()
    return proc.wait()


def run_record(args: argparse.Namespace) -> int:
    root = dataset_root(args)
    if root.exists() and not args.resume:
        print(f"ERROR: dataset already exists at\n  {root}\n")
        print("LeRobot creates it with exist_ok=False, so it refuses to start.")
        print("Pick one:")
        print(f"  --resume                      append episodes to it")
        print(f"  --repo-id local/<new_name>    record a separate dataset")
        print(f"  rm -rf {root}    discard it and start over")
        return 1

    cmd = cmd_record(args)
    print("Cameras:")
    cams = build_cameras((args.cameras or "").split(","), args)
    if not cams:
        print("  (none - recording joint states only)")
    for name, cfg in cams.items():
        target = cfg.get("index_or_path") or cfg.get("serial_number_or_name")
        print(f"  {name:<6} {cfg['type']:<15} {target}  "
              f"{cfg['width']}x{cfg['height']}@{cfg['fps']}")
    print()
    total_s = args.episodes * (args.episode_time_s + args.reset_time_s)
    print("Plan:")
    print(f"  {args.episodes} episodes x {args.episode_time_s:.0f}s recording "
          f"+ {args.reset_time_s:.0f}s reset  ~= {total_s/60:.1f} min total")
    print(f"  dataset rate {args.fps} Hz, cameras at {args.camera_fps} Hz")
    print()
    print("Each episode runs on a TIMER. The sequence is:")
    print("  RECORDING  <- perform the task while this shows")
    print("  RESET      <- put the objects back to the start pose")
    print("  ...repeats until all episodes are done, then it saves.")
    print()
    print("Keyboard shortcuts (right arrow = next, left arrow = re-record,")
    print("escape = stop) are handled by lerobot through pynput, which reads")
    print("the X server rather than this terminal. Over SSH they usually do")
    print("NOT reach it, so treat the timer as the real control and size")
    print("--episode-time-s / --reset-time-s to your task. Ctrl-C always stops.")
    print()
    if args.dry_run:
        print("DRY RUN, command that would run:\n")
        print("  " + " \\\n    ".join(cmd))
        return 0
    # --display_data streams to rerun over the network and does not need X, but
    # keep DISPLAY if the user asked for it in case they are on a local console.
    env = os.environ.copy() if args.display_data else headless_env()
    if args.status:
        print(f"$ {' '.join(cmd)}\n")
        return run_with_status(cmd, args, env=env)
    return run_cli(cmd, env=env)


def primary_ip() -> str:
    """Address of the interface that reaches the outside world (sends nothing)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        sock.close()


def dataset_root(args: argparse.Namespace) -> Path:
    """Where LeRobot will put (or look for) this dataset."""
    if getattr(args, "root", None):
        return Path(args.root)
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        return Path(HF_LEROBOT_HOME) / args.repo_id
    except Exception:
        return Path.home() / ".cache/huggingface/lerobot" / args.repo_id


def run_view(args: argparse.Namespace) -> int:
    """Review a recorded dataset. Headless-friendly via rerun's distant mode."""
    cmd = lerobot_executable("lerobot-dataset-viz")
    cmd += [f"--repo-id={args.repo_id}", f"--episode-index={args.episode}"]
    add_if_value(cmd, "--root", args.root)
    if args.viz_mode == "distant":
        cmd += ["--mode=distant", f"--web-port={args.web_port}",
                f"--grpc-port={args.grpc_port}"]
        ip = primary_ip()
        # The web viewer must be told which gRPC stream to attach to. LeRobot
        # only logs a literal "rerun+http://IP:9876/proxy" placeholder, and the
        # page's built-in address resolves to *the browser's* localhost, so
        # opening the bare page shows rerun's sample data instead of ours.
        stream = urllib.parse.quote(f"rerun+http://{ip}:{args.grpc_port}/proxy",
                                    safe="")
        print("Open this exact URL in a browser on another machine:\n")
        print(f"  http://{ip}:{args.web_port}/?url={stream}\n")
        print("The ?url= part matters. Without it the viewer connects to your")
        print("own machine and shows rerun's example data, not this dataset.\n")
        sys.stdout.flush()   # keep this above the subprocess output
    cmd.extend(args.extra_arg or [])
    return run_cli(cmd)


# ---------------------------------------------------------------------------
# motor bus check
# ---------------------------------------------------------------------------

# SO-101 follower servo IDs, matching lerobot.robots.so_follower.so_follower.
# Physically the arm is numbered bottom (base) to top (gripper) as 1..6.
SO101_MOTOR_IDS = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def run_check_motors(args: argparse.Namespace) -> int:
    """Scan the bus and report which servo IDs answer.

    Use this instead of `lerobot-setup-motors` when the arm was already
    configured elsewhere: motor IDs live in each servo's non-volatile EEPROM, so
    they travel with the hardware. This only reads, it never writes an ID.
    """
    port = args.follower_port
    if not os.path.exists(port):
        print(f"ERROR: {port} does not exist. Is the arm plugged in and powered?")
        return 1
    if not os.access(port, os.R_OK | os.W_OK):
        user = os.environ.get("USER", "your user")
        print(f"ERROR: no read/write permission on {port}.")
        print("  Serial ports are root:dialout and logind grants no ACL for them.")
        print(f"  Fix:  sudo usermod -aG dialout {user}")
        print("  Then log out and back in (or run `newgrp dialout` in this shell).")
        return 1

    from lerobot.motors.feetech import FeetechMotorsBus

    print(f"Scanning {port} at all supported baud rates ...\n")
    found = FeetechMotorsBus.scan_port(port)
    if not found:
        print("\nNo motors responded.")
        print("  - check the arm's power supply (USB alone does not power the servos)")
        print("  - check the daisy-chain cables between servos")
        return 1

    print()
    ok_all = False
    for baudrate, ids in sorted(found.items()):
        print(f"baud {baudrate}: ids {sorted(ids)}")
        expected = set(SO101_MOTOR_IDS)
        got = set(ids)
        for motor_id in sorted(SO101_MOTOR_IDS):
            mark = "ok " if motor_id in got else "MISSING"
            print(f"    id {motor_id}  {SO101_MOTOR_IDS[motor_id]:<14} {mark}")
        extra = got - expected
        if extra:
            print(f"    unexpected ids: {sorted(extra)}")
        if got >= expected:
            ok_all = True

    print()
    if ok_all:
        print("All six SO-101 servos responded. The arm is already configured -")
        print("skip `lerobot-setup-motors` and go straight to calibration:")
        # A leader arm is a *teleoperator* in LeRobot, not a robot, so it is
        # calibrated through --teleop.* ; --robot.type only accepts followers.
        if getattr(args, "role", "follower") == "leader":
            print(f"\n  lerobot-calibrate --teleop.type=so101_leader "
                  f"--teleop.port={port} --teleop.id={args.robot_id}\n")
        else:
            print(f"\n  lerobot-calibrate --robot.type=so101_follower "
                  f"--robot.port={port} --robot.id={args.robot_id}\n")
        return 0
    print("Not all six expected IDs answered. If IDs are wrong or duplicated you do")
    print("need `lerobot-setup-motors`, which rewrites them one motor at a time.")
    return 1


ENCODER_CENTRE = 2047          # int((4096 - 1) / 2), what set_half_turn_homings targets
ENCODER_MAX = 4095
HOMING_MAX_MAGNITUDE = 2047    # Homing_Offset is 11-bit sign-magnitude
WRAP_MARGIN = 250              # how close to 0 / 4095 counts as "near the wrap"
# LeRobot's SO calibrate() treats wrist_roll as a full-turn joint: it is skipped
# by record_ranges_of_motion() and its range is hardcoded to 0..4095.
FULL_TURN_JOINTS = {"wrist_roll"}


def run_center_check(args: argparse.Namespace) -> int:
    """Live encoder readout to position an arm before `lerobot-calibrate`.

    `set_half_turn_homings()` writes `Homing_Offset = present - 2047` into an
    11-bit sign-magnitude field, so the offset only fits while the joint's raw
    encoder is inside one turn. A joint parked on the 0/4095 wrap point can read
    slightly negative (Present_Position is sign-magnitude, sign bit 15) and the
    write then fails with:

        ValueError: Magnitude 2993 exceeds 2047 (max for sign_bit_index=11)

    Nothing here writes to the motors; it only reads.
    """
    port = args.port
    if not os.path.exists(port):
        print(f"ERROR: {port} does not exist.")
        return 1
    if not os.access(port, os.R_OK | os.W_OK):
        print(f"ERROR: no permission on {port}. See Step 1b (dialout group).")
        return 1

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    names = list(SO101_MOTOR_IDS.values())
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
              for i, n in SO101_MOTOR_IDS.items()}
    bus = FeetechMotorsBus(port, motors)
    bus.connect(handshake=False)

    # A calibration run that fails part-way leaves a Homing_Offset written on the
    # motors it already reached. Present_Position is then reported relative to
    # that offset, and reconstructing the raw encoder from it is unreliable (the
    # servo clamps its own output). Zero the offsets first, which is exactly what
    # lerobot-calibrate does as its first step, so Present_Position IS the raw
    # encoder value and the numbers below mean what they say.
    bus.reset_calibration()
    print("Homing offsets reset to 0 (same first step as lerobot-calibrate),")
    print("so the values below are true raw encoder counts.\n")

    print(f"Live encoder readout on {port}.")
    print(f"Target is the encoder centre ({ENCODER_CENTRE}); rows go bad near the")
    print("0/4095 wrap because the homing offset would overflow.")
    print()
    print("Park every joint near mid-travel, then run lerobot-calibrate.")
    print()
    print("Do NOT sweep joints here. These servos keep a multi-turn counter in RAM:")
    print("once a joint is driven past 0 or 4095 the reading keeps counting (e.g.")
    print("5079) instead of wrapping, and stays that way until the arm is power")
    print("cycled - which is exactly what makes set_half_turn_homings() overflow.")
    print("Sweep the joints later, when lerobot-calibrate asks you to; by then the")
    print("homing offsets have re-centred everything on 2047 and it is safe.")
    print()
    print("If a value is already out of range, power cycle the arm to clear it.")
    print()
    print("Ctrl-C to stop.\n")

    tty_out = sys.stdout.isatty()
    lines = 0
    seen_min: dict[str, int] = {}
    seen_max: dict[str, int] = {}
    last_raw: dict[str, int] = {}
    wrapped: set[str] = set()
    try:
        while True:
            rows = []
            all_ok = True
            for name in names:
                try:
                    # offsets were zeroed at startup, so this IS the raw encoder
                    actual = bus.read("Present_Position", name, normalize=False)

                    prev = last_raw.get(name)
                    if prev is not None and abs(actual - prev) > 2000:
                        # a real joint cannot jump ~175 deg between samples
                        wrapped.add(name)
                    last_raw[name] = actual
                    seen_min[name] = min(seen_min.get(name, actual), actual)
                    seen_max[name] = max(seen_max.get(name, actual), actual)

                    needed = actual - ENCODER_CENTRE
                    lo, hi = seen_min[name], seen_max[name]
                    # LeRobot excludes wrist_roll from record_ranges_of_motion and
                    # hardcodes its range to 0..4095, so how far it was swept does
                    # not matter. Only its position when you press ENTER matters,
                    # because set_half_turn_homings() still runs on every motor.
                    if name in FULL_TURN_JOINTS:
                        lo = hi = actual
                    # A joint whose travel dips below 0 or above 4095 cannot be
                    # calibrated at all: Homing_Offset is capped at +-2047, so the
                    # whole sweep must live inside one turn. Feetech reports
                    # sub-zero positions as sign-magnitude negatives rather than
                    # wrapping to 4095, so check the sign explicitly - a jump-based
                    # wrap test alone silently misses this.
                    if name in wrapped:
                        status, ok = "WRAPPED - re-seat horn", False
                    elif lo < 0 or hi > ENCODER_MAX:
                        # NOT necessarily a hardware fault. The sweep inflated the
                        # multi-turn counter; a power cycle with the joint at
                        # mid-travel puts raw back inside 0..4095, and then the
                        # offset always fits. Re-seating is only needed if a joint
                        # still cannot be parked legally after a power cycle.
                        status, ok = "swept past a turn - power cycle at mid-travel", False
                    elif actual < 0 or actual > ENCODER_MAX:
                        status, ok = "OFF-SCALE", False
                    elif abs(needed) > HOMING_MAX_MAGNITUDE:
                        status, ok = "OUT OF RANGE", False
                    elif actual < WRAP_MARGIN or actual > ENCODER_MAX - WRAP_MARGIN:
                        status, ok = "NEAR WRAP", False
                    else:
                        status, ok = "OK", True
                    all_ok &= ok
                    rows.append("  %-14s raw=%6d  offset=%+6d  seen=%5d..%-5d %s"
                                % (name, actual, needed, lo, hi, status))
                    if not ok and name in FULL_TURN_JOINTS:
                        rows.append("  %-14s   free-spinning: turn by hand toward %d"
                                    % ("", ENCODER_CENTRE))
                    elif not ok and (lo < 0 or hi > ENCODER_MAX):
                        centre = (lo + hi) // 2
                        shift = ENCODER_CENTRE - centre
                        # rotation is modulo one turn, so report the smallest
                        # equivalent angle rather than e.g. -260 deg for +100
                        while shift > 2048:
                            shift -= 4096
                        while shift < -2048:
                            shift += 4096
                        rows.append("  %-14s   span %d; park mid-travel + power "
                                    "cycle. Only if that fails: rotate horn "
                                    "%+d (%+.0f deg)"
                                    % ("", hi - lo, shift, shift * 360.0 / 4096))
                except Exception as exc:
                    all_ok = False
                    rows.append("  %-14s read error: %s" % (name, exc))

            verdict = ("  ALL CENTRED - safe to run lerobot-calibrate"
                       if all_ok else
                       "  NOT READY - move the flagged joints toward mid-travel")
            # advice lines make the frame height vary, which breaks the
            # cursor-up redraw and smears rows; pad to a fixed height instead
            out = rows + [""] * (2 * len(names) - len(rows)) + ["", verdict]
            if tty_out:
                # A line longer than the terminal wraps onto a second physical
                # row, so cursor-up by the logical line count lands in the wrong
                # place and the rows smear. Truncate to the real width.
                width = max(40, shutil.get_terminal_size((100, 24)).columns - 1)
                out = [line[:width] for line in out]
                if lines:
                    sys.stdout.write(f"\033[{lines}A")
                for line in out:
                    sys.stdout.write("\033[2K" + line + "\n")
                lines = len(out)
                sys.stdout.flush()
            else:
                print("\n".join(out) + "\n")
            time.sleep(0.3 if tty_out else 2.0)
    except KeyboardInterrupt:
        print()
    finally:
        bus.disconnect()
    return 0


# ---------------------------------------------------------------------------
# PS5 DualSense connected directly to the Jetson (evdev, no display needed)
# ---------------------------------------------------------------------------

# This Jetson kernel (5.15-tegra) has no hid_playstation driver, so a DualSense
# enumerates as a generic HID gamepad: HID buttons 1..15 land on the generic
# BTN_SOUTH..BTN_THUMBR block in report order. Verified on cmpe-jetson with
# jetson_devices.py: Circle -> BTN_C(306), Triangle -> BTN_NORTH(307),
# L2 button -> BTN_TL(310), R2 button -> BTN_TR(311).
PS5_BUTTON_CODES = {
    "square": 304,
    "cross": 305,
    "circle": 306,
    "triangle": 307,
    "l1": 308,
    "r1": 309,
    "l2_button": 310,
    "r2_button": 311,
    "create": 312,
    "options": 313,
    "l3": 314,
    "r3": 315,
    "ps": 316,
    "touchpad": 317,
    "mute": 318,
}
PS5_BUTTON_NAMES = {code: name for name, code in PS5_BUTTON_CODES.items()}

# Axis codes confirmed from absinfo on this host: ABS_RX/ABS_RY rest at 0 (they
# are the analog triggers) while ABS_Z/ABS_RZ rest at mid-scale (right stick).
PS5_AXIS_STICK = {0: "lx", 1: "ly", 2: "rx", 5: "ry"}
PS5_AXIS_TRIGGER = {3: "l2", 4: "r2"}
PS5_AXIS_HAT = {16: "hat_x", 17: "hat_y"}


class DualSenseReader:
    """Reads a DualSense through evdev and exposes normalized state.

    Sticks are reported as -1..1, triggers as 0..1, the hat as -1/0/1.
    Reading runs on a background thread so the control loop never blocks.
    """

    def __init__(self, device_path: str | None = None) -> None:
        try:
            import evdev
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "evdev is required for ps5-local. Install it in this environment:\n"
                "  ~/.local/bin/uv pip install --python $(which python) evdev"
            ) from exc

        self.evdev = evdev
        self.device = self._open_device(device_path)
        self.absinfo: dict[int, Any] = {}
        caps = self.device.capabilities(absinfo=True)
        for code, info in caps.get(evdev.ecodes.EV_ABS, []):
            self.absinfo[code] = info

        self.lock = threading.Lock()
        self.axes: dict[str, float] = {
            "lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0,
            "l2": 0.0, "r2": 0.0, "hat_x": 0.0, "hat_y": 0.0,
        }
        self.buttons: set[int] = set()
        self.last_event = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open_device(self, device_path: str | None):
        evdev = self.evdev
        if device_path:
            return evdev.InputDevice(device_path)
        candidates = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            name = dev.name.lower()
            if "dualsense" in name or "wireless controller" in name:
                candidates.append(dev)
            else:
                dev.close()
        if not candidates:
            raise RuntimeError(
                "No DualSense found on /dev/input/event*.\n"
                "Pair and connect it first:\n"
                "  python ~/jetson_devices.py bluetooth ps5\n"
                "If it is connected but invisible here, the event node may not be "
                "readable by this user."
            )
        for extra in candidates[1:]:
            extra.close()
        return candidates[0]

    def _normalize(self, code: int, value: int) -> tuple[str, float] | None:
        info = self.absinfo.get(code)
        lo, hi = (info.min, info.max) if info else (0, 255)
        if hi == lo:
            return None
        if code in PS5_AXIS_HAT:
            return PS5_AXIS_HAT[code], float(value)
        if code in PS5_AXIS_TRIGGER:
            return PS5_AXIS_TRIGGER[code], (value - lo) / (hi - lo)
        if code in PS5_AXIS_STICK:
            mid = (hi + lo) / 2.0
            norm = (value - mid) / ((hi - lo) / 2.0)
            return PS5_AXIS_STICK[code], 0.0 if abs(norm) < 0.01 else norm
        return None

    def _run(self) -> None:
        ecodes = self.evdev.ecodes
        try:
            for event in self.device.read_loop():
                if self._stop.is_set():
                    return
                with self.lock:
                    self.last_event = time.monotonic()
                    if event.type == ecodes.EV_KEY:
                        if event.value == 1:
                            self.buttons.add(event.code)
                        elif event.value == 0:
                            self.buttons.discard(event.code)
                    elif event.type == ecodes.EV_ABS:
                        parsed = self._normalize(event.code, event.value)
                        if parsed is not None:
                            self.axes[parsed[0]] = parsed[1]
        except OSError as exc:
            # close() closes the fd while this thread is blocked in read_loop(),
            # so an EBADF here during shutdown is us, not the controller.
            if not self._stop.is_set():
                LOG.error("Controller disconnected: %s", exc)
            else:
                LOG.debug("reader thread stopped: %s", exc)
            self._stop.set()

    @property
    def alive(self) -> bool:
        return not self._stop.is_set()

    def snapshot(self) -> tuple[dict[str, float], set[int]]:
        with self.lock:
            return dict(self.axes), set(self.buttons)

    def pressed(self, name: str) -> bool:
        code = PS5_BUTTON_CODES.get(name)
        if code is None:
            return False
        with self.lock:
            return code in self.buttons

    def close(self) -> None:
        self._stop.set()
        try:
            self.device.close()
        except Exception:
            pass


PS5_CONTROL_HELP = """
  Left stick   left/right   shoulder_pan
  Left stick   up/down      shoulder_lift
  Right stick  up/down      elbow_flex
  Right stick  left/right   wrist_flex
  D-pad        left/right   wrist_roll
  R2 trigger                open gripper
  L2 trigger                close gripper

  L1  (hold)                DEADMAN - the arm only moves while this is held
  Circle                    E-STOP  - freeze until you press Options
  Options                   clear E-STOP
  PS                        quit
"""


def ps5_axes_to_delta(
    axes: dict[str, float],
    dt_s: float,
    joint_speed_deg_s: float,
    wrist_roll_speed_deg_s: float,
    gripper_speed_s: float,
    deadzone: float,
) -> dict[str, float]:
    """Map normalized DualSense axes onto per-joint deltas for this timestep."""
    lx = apply_deadzone(axes.get("lx", 0.0), deadzone)
    ly = apply_deadzone(axes.get("ly", 0.0), deadzone)
    rx = apply_deadzone(axes.get("rx", 0.0), deadzone)
    ry = apply_deadzone(axes.get("ry", 0.0), deadzone)
    hat_x = axes.get("hat_x", 0.0)
    l2 = axes.get("l2", 0.0)
    r2 = axes.get("r2", 0.0)

    return {
        "shoulder_pan.pos": lx * joint_speed_deg_s * dt_s,
        "shoulder_lift.pos": -ly * joint_speed_deg_s * dt_s,
        "elbow_flex.pos": -ry * joint_speed_deg_s * dt_s,
        "wrist_flex.pos": rx * joint_speed_deg_s * dt_s,
        "wrist_roll.pos": hat_x * wrist_roll_speed_deg_s * dt_s,
        "gripper.pos": (r2 - l2) * gripper_speed_s * dt_s,
    }


def run_ps5_local(args: argparse.Namespace) -> int:
    """Drive the SO-ARM101 from a DualSense connected directly to the Jetson."""
    limits = parse_limits(args.limits_json)
    pad = DualSenseReader(args.device)
    print(f"Controller: {pad.device.name}  ({pad.device.path})")
    print(PS5_CONTROL_HELP)

    robot = None if args.dry_run else make_so101_robot(args)
    if robot is not None:
        robot.connect(calibrate=not args.no_calibrate)
        goal = observation_to_goal(robot.get_observation())
    else:
        print("DRY RUN: no robot connected, printing goals only.\n")
        goal = {key: 0.0 for key in MOTOR_KEYS}

    # Stay strictly inside max_relative_target so the goal never parks exactly on
    # LeRobot's clamp boundary (which is what makes the warning repeat forever).
    lead_band = args.goal_lead_deg
    if lead_band is None:
        lead_band = max(1.0, 0.8 * float(args.max_relative_target or 8.0))

    period = 1.0 / max(args.fps, 1)
    last_loop = time.monotonic()
    last_print = 0.0
    last_idle_read = 0.0
    estop = False
    try:
        while True:
            now = time.monotonic()
            dt_s = min(max(now - last_loop, 1e-3), 0.2)
            last_loop = now

            if not pad.alive:
                print("\nController disconnected. Stopping.")
                return 1

            axes, buttons = pad.snapshot()
            if PS5_BUTTON_CODES["ps"] in buttons:
                print("\nPS pressed. Exiting.")
                return 0
            if PS5_BUTTON_CODES["circle"] in buttons:
                estop = True
            if PS5_BUTTON_CODES["options"] in buttons:
                estop = False

            deadman = PS5_BUTTON_CODES[args.deadman_button] in buttons
            moving = deadman and not estop

            # Anti-windup: this is an integrating controller, so if a joint stalls
            # against a mechanical or calibration limit the goal keeps marching past
            # the reachable position. It then sits permanently outside
            # max_relative_target and LeRobot logs "Relative goal position magnitude
            # had to be clamped to be safe" on every cycle. Re-reading the measured
            # position and holding the goal inside a band around it keeps the goal
            # honest, and also stops the arm lurching when the deadman is re-pressed.
            # send_action() already sync_reads present position when
            # max_relative_target is set, so this is a second read. Take it every
            # cycle while moving (the clamp needs it) but only occasionally while
            # idle, where it just keeps the goal tracking reality.
            present: dict[str, float] | None = None
            need_read = moving or (now - last_idle_read) > 0.2
            if robot is not None and need_read:
                try:
                    present = observation_to_goal(robot.get_observation())
                    if not moving:
                        last_idle_read = now
                except Exception as exc:
                    LOG.debug("observation read failed: %s", exc)

            if moving:
                delta = ps5_axes_to_delta(
                    axes=axes,
                    dt_s=dt_s,
                    joint_speed_deg_s=args.joint_speed_deg_s,
                    wrist_roll_speed_deg_s=args.wrist_roll_speed_deg_s,
                    gripper_speed_s=args.gripper_speed_s,
                    deadzone=args.deadzone,
                )
                for key, value in delta.items():
                    goal[key] += value
                apply_limits(goal, limits)
                if present is not None:
                    for key in goal:
                        if key in present:
                            goal[key] = clamp(goal[key],
                                              present[key] - lead_band,
                                              present[key] + lead_band)
                if robot is not None:
                    sent = robot.send_action(goal)
                    goal.update({k: float(v) for k, v in sent.items() if k in goal})
            elif present is not None:
                # idle: track the real position so re-engaging never jumps
                goal.update(present)

            if now - last_print > args.status_period_s:
                state = "ESTOP" if estop else ("MOVING" if moving else "HOLD")
                summary = " ".join(
                    f"{key.split('.')[0]}={goal[key]:7.2f}" for key in MOTOR_KEYS
                )
                held = ",".join(sorted(PS5_BUTTON_NAMES.get(b, str(b)) for b in buttons))
                line = f"{state:6s} {summary}  held=[{held}]"
                if sys.stdout.isatty():
                    print(f"\r{line:<160}", end="", flush=True)
                else:
                    print(line, flush=True)
                last_print = now

            time.sleep(max(period - (time.monotonic() - now), 0.0))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        pad.close()
        if robot is not None:
            robot.disconnect()


# ---------------------------------------------------------------------------
# interactive guided menu
# ---------------------------------------------------------------------------

def detect_serial_ports() -> list[str]:
    ports = sorted(Path("/dev").glob("ttyACM*")) + sorted(Path("/dev").glob("ttyUSB*"))
    return [str(p) for p in ports]


def run_cli(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call(cmd, env=env)


def headless_env() -> dict:
    """Environment with DISPLAY removed.

    LeRobot decides whether to use pynput by trying to import it, and pynput
    imports fine whenever DISPLAY is set - including an SSH-forwarded display
    that does not support the X11 RECORD extension. It then builds a listener
    that immediately dies with `AttributeError: record_create_context`. Dropping
    DISPLAY makes that check honest, so pynput is skipped entirely.
    """
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    return env


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{text}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


def run_interactive(args: argparse.Namespace) -> int:
    """Guided menu that walks through bring-up in the order things should be done."""
    state = argparse.Namespace(
        follower_port=args.follower_port,
        leader_port=args.leader_port,
        robot_id=args.robot_id,
        leader_id=args.leader_id,
        calibration_dir=None,
        leader_calibration_dir=None,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        max_relative_target=args.max_relative_target,
        no_calibrate=True,
        limits_json=None,
        fps=args.fps,
        dry_run=False,
        device=None,
        deadman_button="l1",
        deadzone=0.08,
        joint_speed_deg_s=args.joint_speed_deg_s,
        wrist_roll_speed_deg_s=45.0,
        gripper_speed_s=35.0,
        status_period_s=0.2,
        goal_lead_deg=None,
        joint_step_deg=2.0,
        gripper_step=2.0,
        role="follower",
    )

    while True:
        ports = detect_serial_ports()
        print("\n" + "=" * 66)
        print("  SO-ARM101 teleop - interactive")
        print("=" * 66)
        print(f"  follower port : {state.follower_port or '(not set)'}")
        print(f"  leader port   : {state.leader_port or '(not set)'}")
        print(f"  robot id      : {state.robot_id}")
        print(f"  serial ports  : {', '.join(ports) if ports else '(none detected)'}")
        print()
        print("  1. detect serial ports")
        print("  2. set follower port")
        print("  3. lerobot-find-port  (unplug/replug to identify an arm)")
        print("  4. lerobot-info       (environment check)")
        print("  5. calibrate follower (lerobot-calibrate)")
        print("  5b. check motor IDs   (read-only bus scan)")
        print("  6. keyboard jogging test")
        print("  7. PS5 controller status / pairing")
        print("  8. PS5 local teleop   (dry run, no robot)")
        print("  9. PS5 local teleop   (real robot)")
        print("  10. leader-arm teleop (lerobot-teleoperate)")
        print("  11. record a dataset  (lerobot-record, wrist camera)")
        print("  12. review a dataset  (lerobot-dataset-viz)")
        print("  q. quit")

        choice = prompt("\n  action").lower()
        if choice in {"q", "quit", "exit"}:
            return 0

        try:
            if choice == "1":
                if ports:
                    for port in ports:
                        print(f"    {port}")
                else:
                    print("    No /dev/ttyACM* or /dev/ttyUSB* found.")
                    print("    Plug the SO-ARM101 controller board in via USB and retry.")
            elif choice == "2":
                default = ports[0] if ports else ""
                state.follower_port = prompt("  follower port", default)
            elif choice == "3":
                run_cli(lerobot_executable("lerobot-find-port"))
            elif choice == "4":
                run_cli(lerobot_executable("lerobot-info"))
            elif choice == "5":
                if not state.follower_port:
                    print("  Set the follower port first (option 2).")
                    continue
                cmd = lerobot_executable("lerobot-calibrate") + [
                    "--robot.type=so101_follower",
                    f"--robot.port={state.follower_port}",
                    f"--robot.id={state.robot_id}",
                ]
                run_cli(cmd)
            elif choice == "5b":
                if not state.follower_port:
                    print("  Set the follower port first (option 2).")
                    continue
                run_check_motors(state)
            elif choice == "6":
                if not state.follower_port:
                    print("  Set the follower port first (option 2).")
                    continue
                run_keyboard(state)
            elif choice in {"7"}:
                tool = Path.home() / "jetson_devices.py"
                if not tool.exists():
                    tool = Path(__file__).resolve().parent / "jetson_devices.py"
                run_cli([sys.executable, str(tool), "bluetooth", "ps5"])
            elif choice in {"8", "9"}:
                if choice == "9" and not state.follower_port:
                    print("  Set the follower port first (option 2).")
                    continue
                state.dry_run = choice == "8"
                run_ps5_local(state)
                state.dry_run = False
            elif choice == "10":
                if not state.follower_port:
                    print("  Set the follower port first (option 2).")
                    continue
                if not state.leader_port:
                    default = next((p for p in ports if p != state.follower_port), "")
                    state.leader_port = prompt("  leader port", default)
                if not state.leader_port:
                    continue
                leader_args = argparse.Namespace(
                    **vars(state),
                    time_s=None,
                    display_data=False,
                    extra_arg=None,
                )
                run_lerobot_teleoperate(leader_args, "so101_leader")
            else:
                print("  Unknown action.")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as exc:  # keep the menu alive on operator error
            print(f"  ERROR: {type(exc).__name__}: {exc}")


def run_api_post(args: argparse.Namespace) -> int:
    payload = load_json_arg(args.json, {})
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
        print(response.read().decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_robot_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--follower-port", required=True, help="SO-ARM101 follower serial port on the Jetson.")
        p.add_argument("--robot-id", default="so101_follower")
        p.add_argument("--calibration-dir", default=None)
        p.add_argument("--use-degrees", type=parse_bool, default=True)
        p.add_argument("--disable-torque-on-disconnect", type=parse_bool, default=True)
        p.add_argument("--max-relative-target", type=parse_max_relative_target,
                       default=8.0,
                       help="Cap on how far one command may move a joint. "
                            "'none' disables it (LeRobot's own default).")
        p.add_argument("--no-calibrate", action="store_true", help="Do not enter calibration flow on connect.")

    leader = sub.add_parser("leader", help="Use a physical SO-ARM101 leader arm via LeRobot CLI.")
    add_robot_args(leader)
    leader.add_argument("--leader-port", required=True)
    leader.add_argument("--leader-id", default="so101_leader")
    leader.add_argument("--leader-calibration-dir", default=None)
    leader.add_argument("--fps", type=int, default=60)
    leader.add_argument("--time-s", type=float, default=None)
    leader.add_argument("--display-data", type=parse_bool, default=False)
    leader.add_argument("--extra-arg", action="append", help="Raw extra arg passed to lerobot-teleoperate.")
    # The follower trails the leader by several degrees under load, so an 8 deg
    # cap clamps almost continuously while you move. Roomier default here.
    leader.set_defaults(max_relative_target=30.0)

    check = sub.add_parser(
        "check-motors",
        help="Scan the servo bus and report which SO-101 motor IDs answer (read-only).",
    )
    check.add_argument("--follower-port", required=True,
                       help="Serial port of the arm to scan (follower or leader).")
    check.add_argument("--robot-id", default="so101_follower")
    check.add_argument("--role", choices=["follower", "leader"], default="follower",
                       help="Which calibrate command to suggest. A leader arm is a "
                            "LeRobot *teleoperator*, so it uses --teleop.* flags.")

    centre = sub.add_parser(
        "center-check",
        help="Live encoder readout: position an arm safely before lerobot-calibrate.",
    )
    centre.add_argument("--port", required=True, help="Serial port of the arm.")

    rec = sub.add_parser(
        "record",
        help="Record a LeRobot dataset with leader teleop and optional cameras.",
    )
    add_robot_args(rec)
    rec.add_argument("--leader-port", required=True)
    rec.add_argument("--leader-id", default="so101_leader")
    rec.add_argument("--repo-id", required=True,
                     help="Dataset id, e.g. myuser/so101_pick. Any 'a/b' works locally.")
    rec.add_argument("--task", required=True, help="Natural-language task description.")
    rec.add_argument("--cameras", default="wrist",
                     help="Comma separated presets: wrist, scene, or none. "
                          "wrist = USB UVC on the arm, scene = RealSense "
                          "(desk is accepted as an alias for scene).")
    rec.add_argument("--wrist-device", default=None, help="Override /dev/videoN for wrist.")
    rec.add_argument("--wrist-fourcc", default="MJPG",
                     help="FOURCC for the wrist camera. MJPG keeps the frame rate up.")
    rec.add_argument("--wrist-rotation", type=int, default=0,
                     choices=[0, 90, 180, 270], help="Rotate the wrist image.")
    rec.add_argument("--scene-serial", "--desk-serial", dest="desk_serial",
                     default=None, help="Override the RealSense serial for 'scene'.")
    rec.add_argument("--width", type=int, default=640)
    rec.add_argument("--height", type=int, default=480)
    rec.add_argument("--fps", type=int, default=30,
                     help="Dataset / control-loop rate. 30 works because "
                          "get_observation() uses the non-blocking "
                          "read_latest(); only a blocking read() would overrun.")
    rec.add_argument("--camera-fps", type=int, default=30,
                     help="Camera hardware rate. Must be a mode the camera "
                          "advertises (this UVC camera only offers 30).")
    rec.add_argument("--episodes", type=int, default=5)
    rec.add_argument("--episode-time-s", type=float, default=20.0)
    rec.add_argument("--reset-time-s", type=float, default=5.0)
    rec.add_argument("--root", default=None, help="Local dataset directory.")
    rec.add_argument("--push-to-hub", type=parse_bool, default=False,
                     help="Upload to the Hub. Off by default so no login is needed.")
    rec.add_argument("--vcodec", default="libsvtav1",
                     help="Video codec. Benchmarked on this Jetson, libsvtav1 "
                          "beats libx264 on both speed and size (1.3s vs 2.7s "
                          "for 200 frames), so keep LeRobot's default.")
    rec.add_argument("--streaming-encoding", type=parse_bool, default=False,
                     help="Encode while recording instead of after each episode. "
                          "Removes the between-episode pause.")
    rec.add_argument("--encoder-threads", type=int, default=None)
    rec.add_argument("--resume", action="store_true", help="Append to an existing dataset.")
    rec.add_argument("--display-data", action="store_true")
    rec.add_argument("--dry-run", action="store_true",
                     help="Print the lerobot-record command without running it.")
    rec.add_argument("--extra-arg", action="append")
    rec.add_argument("--status", type=parse_bool, default=True,
                     help="Show a live episode/phase panel. Set false for raw "
                          "lerobot-record output.")
    rec.set_defaults(max_relative_target=30.0)

    view = sub.add_parser("view", aliases=["dataset-viz"],
                          help="Review a recorded dataset (rerun, browser-friendly).")
    view.add_argument("--repo-id", required=True)
    view.add_argument("--episode", type=int, default=0)
    view.add_argument("--root", default=None)
    view.add_argument("--mode", dest="viz_mode", choices=["local", "distant"],
                      default="distant",
                      help="distant serves rerun over the network for a headless "
                           "Jetson. Stored as viz_mode because the subparsers "
                           "already use dest='mode' for the subcommand name.")
    view.add_argument("--web-port", type=int, default=9090)
    view.add_argument("--grpc-port", type=int, default=9876)
    view.add_argument("--extra-arg", action="append")

    ps5 = sub.add_parser(
        "ps5-local",
        help="Drive the arm from a DualSense connected directly to the Jetson.",
    )
    add_robot_args(ps5)
    ps5.add_argument("--device", default=None,
                     help="/dev/input/eventN of the DualSense; autodetected if omitted.")
    ps5.add_argument("--fps", type=int, default=50)
    ps5.add_argument("--deadzone", type=float, default=0.08)
    ps5.add_argument("--joint-speed-deg-s", type=float, default=30.0)
    ps5.add_argument("--wrist-roll-speed-deg-s", type=float, default=45.0)
    ps5.add_argument("--gripper-speed-s", type=float, default=35.0)
    ps5.add_argument("--deadman-button", default="l1", choices=sorted(PS5_BUTTON_CODES),
                     help="Button that must be held for the arm to move.")
    ps5.add_argument("--limits-json", default=None, help="JSON dict motor->[min,max], or path.")
    ps5.add_argument("--status-period-s", type=float, default=0.2)
    ps5.add_argument("--goal-lead-deg", type=float, default=None,
                     help="How far the goal may lead the measured position before "
                          "being held back (anti-windup). Default: 0.8 x "
                          "--max-relative-target.")
    ps5.add_argument("--dry-run", action="store_true",
                     help="Read the controller and print goals without moving the arm.")

    interactive = sub.add_parser(
        "interactive",
        help="Guided menu: find port, calibrate, keyboard test, PS5 test, leader teleop.",
    )
    interactive.add_argument("--follower-port", default=None)
    interactive.add_argument("--leader-port", default=None)
    interactive.add_argument("--robot-id", default="so101_follower")
    interactive.add_argument("--leader-id", default="so101_leader")
    interactive.add_argument("--fps", type=int, default=50)
    interactive.add_argument("--max-relative-target", type=float, default=8.0)
    interactive.add_argument("--joint-speed-deg-s", type=float, default=30.0)

    gamepad = sub.add_parser(
        "gamepad-local",
        help="LeRobot's gamepad teleoperator (end-effector deltas; needs an IK robot, "
             "not a plain so101_follower - prefer ps5-local).",
    )
    add_robot_args(gamepad)
    gamepad.add_argument("--fps", type=int, default=60)
    gamepad.add_argument("--time-s", type=float, default=None)
    gamepad.add_argument("--display-data", type=parse_bool, default=False)
    gamepad.add_argument("--extra-arg", action="append", help="Raw extra arg passed to lerobot-teleoperate.")

    keyboard = sub.add_parser("keyboard", help="Terminal joint jogging without X11/pynput.")
    add_robot_args(keyboard)
    keyboard.add_argument("--fps", type=int, default=30)
    keyboard.add_argument("--joint-step-deg", type=float, default=2.0)
    keyboard.add_argument("--gripper-step", type=float, default=2.0)
    keyboard.add_argument("--limits-json", default=None, help="JSON dict motor->[min,max], or path.")

    remote = sub.add_parser("remote-server", help="Jetson UDP/HTTP server for remote commands.")
    add_robot_args(remote)
    remote.add_argument("--bind-host", default="0.0.0.0")
    remote.add_argument("--udp-port", type=int, default=8766)
    remote.add_argument("--http-port", type=int, default=8765)
    remote.add_argument("--fps", type=int, default=50)
    remote.add_argument("--command-timeout-s", type=float, default=0.4)
    remote.add_argument("--require-deadman", type=parse_bool, default=True)
    remote.add_argument("--deadzone", type=float, default=0.08)
    remote.add_argument("--joint-speed-deg-s", type=float, default=30.0)
    remote.add_argument("--wrist-roll-speed-deg-s", type=float, default=45.0)
    remote.add_argument("--gripper-speed-s", type=float, default=35.0)
    remote.add_argument("--limits-json", default=None, help="JSON dict motor->[min,max], or path.")
    remote.add_argument("--status-period-s", type=float, default=1.0)
    remote.add_argument("--dry-run", action="store_true")

    mac = sub.add_parser("mac-ps5-client", help="Mac client that sends PS5 joystick state to remote-server.")
    mac.add_argument("--jetson-host", required=True)
    mac.add_argument("--udp-port", type=int, default=8766)
    mac.add_argument("--fps", type=int, default=50)
    mac.add_argument("--joystick-index", type=int, default=0)
    mac.add_argument("--axis-lx", type=int, default=0)
    mac.add_argument("--axis-ly", type=int, default=1)
    mac.add_argument("--axis-rx", type=int, default=2)
    mac.add_argument("--axis-ry", type=int, default=3)
    mac.add_argument("--axis-l2", type=int, default=4)
    mac.add_argument("--axis-r2", type=int, default=5)
    mac.add_argument("--scale-lx", type=float, default=1.0)
    mac.add_argument("--scale-ly", type=float, default=1.0)
    mac.add_argument("--scale-rx", type=float, default=1.0)
    mac.add_argument("--scale-ry", type=float, default=1.0)
    mac.add_argument("--deadman-button", type=int, default=5, help="Set -1 to always send deadman=true.")
    mac.add_argument("--estop-button", type=int, default=1, help="Set -1 to disable.")
    mac.add_argument("--wrist-negative-button", type=int, default=-1)
    mac.add_argument("--wrist-positive-button", type=int, default=-1)
    mac.add_argument("--print-events", action="store_true")

    post = sub.add_parser("api-post", help="POST one JSON command to a remote-server.")
    post.add_argument("--url", default="http://jetsonorin:8765/command")
    post.add_argument("--json", required=True, help="Inline JSON payload or path to a JSON file.")
    post.add_argument("--timeout-s", type=float, default=3.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.mode == "leader":
        return run_lerobot_teleoperate(args, "so101_leader")
    if args.mode == "gamepad-local":
        return run_lerobot_teleoperate(args, "gamepad")
    if args.mode == "keyboard":
        return run_keyboard(args)
    if args.mode == "record":
        return run_record(args)
    if args.mode in {"view", "dataset-viz"}:
        return run_view(args)
    if args.mode == "check-motors":
        return run_check_motors(args)
    if args.mode == "center-check":
        return run_center_check(args)
    if args.mode == "ps5-local":
        return run_ps5_local(args)
    if args.mode == "interactive":
        return run_interactive(args)
    if args.mode == "remote-server":
        return run_remote_server(args)
    if args.mode == "mac-ps5-client":
        return run_mac_ps5_client(args)
    if args.mode == "api-post":
        return run_api_post(args)
    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
