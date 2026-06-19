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
import json
import logging
import math
import os
import queue
import select
import shutil
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
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
        p.add_argument("--max-relative-target", type=float, default=8.0)
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

    gamepad = sub.add_parser("gamepad-local", help="Use LeRobot's local gamepad teleoperator on the Jetson.")
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
