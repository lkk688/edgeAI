#!/usr/bin/env python3
"""
jetson_devices.py - all-in-one device detection and test CLI for Jetson Orin Nano.

Two command groups:

  camera     detect / describe / capture / record / view cameras
             (CSI IMX219 via Argus, USB UVC, Intel RealSense, Orbbec)
  bluetooth  detect / pair / connect / test Bluetooth controllers
             (Sony PS5 DualSense, Meta Quest 3 Touch Plus, generic gamepads)

Examples:

  ./jetson_devices.py camera list --formats
  ./jetson_devices.py camera test
  ./jetson_devices.py camera snap --camera csi:0 --out /tmp/csi0.jpg
  ./jetson_devices.py camera record --camera usb --duration 5
  ./jetson_devices.py camera show --camera rs:depth --mode stream --port 8090
  ./jetson_devices.py camera interactive

  ./jetson_devices.py bluetooth status
  ./jetson_devices.py bluetooth scan --timeout 15
  ./jetson_devices.py bluetooth ps5
  ./jetson_devices.py bluetooth quest
  ./jetson_devices.py bluetooth gamepad
  ./jetson_devices.py bluetooth interactive

  ./jetson_devices.py doctor

The script is standard-library only for its core paths. Optional features
(RealSense SDK streams, gamepad event reading) auto-locate a Python
interpreter that has the needed module and re-exec into it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterator, Optional

VERSION = "1.0.0"
DEFAULT_OUTDIR = os.path.expanduser("~/camera_samples")

# Interpreters searched when an optional module is missing from the current one.
CANDIDATE_PYTHONS = [
    os.path.expanduser("~/lerobot-py310-cuda/bin/python"),
    os.path.expanduser("~/.venv/bin/python"),
    os.path.expanduser("~/lerobot-py312/bin/python"),
    "/usr/bin/python3",
]

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def c(text: str, color: str) -> str:
    if _NO_COLOR:
        return text
    codes = {
        "red": "31", "green": "32", "yellow": "33", "blue": "34",
        "magenta": "35", "cyan": "36", "grey": "90", "bold": "1",
    }
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def hdr(text: str) -> None:
    print()
    print(c(text, "bold"))
    print(c("-" * len(text), "grey"))


def ok(text: str) -> None:
    print(f"  {c('[ OK ]', 'green')} {text}")


def warn(text: str) -> None:
    print(f"  {c('[WARN]', 'yellow')} {text}")


def fail(text: str) -> None:
    print(f"  {c('[FAIL]', 'red')} {text}")


def info(text: str) -> None:
    print(f"  {c('[INFO]', 'cyan')} {text}")


def run(cmd: list[str] | str, timeout: int = 20, env: Optional[dict] = None) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr). Never raises on failure."""
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True,
            timeout=timeout, env=env or os.environ.copy(),
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def require_module(mod: str, why: str) -> Any:
    """Import `mod`; if missing, re-exec this script under an interpreter that has it."""
    try:
        return __import__(mod)
    except ImportError:
        pass
    if os.environ.get("JETSON_DEVICES_REEXEC"):
        fail(f"module '{mod}' is required for {why} but no interpreter provides it")
        print(f"       tried: {', '.join(CANDIDATE_PYTHONS)}")
        sys.exit(3)
    for py in CANDIDATE_PYTHONS:
        if not os.path.exists(py) or os.path.realpath(py) == os.path.realpath(sys.executable):
            continue
        rc, _, _ = run([py, "-c", f"import {mod}"], timeout=60)
        if rc == 0:
            env = os.environ.copy()
            env["JETSON_DEVICES_REEXEC"] = "1"
            ld = env.get("LD_LIBRARY_PATH", "")
            cudss = os.path.expanduser("~/lerobot-py310-cuda/cudss-lib")
            if os.path.isdir(cudss) and cudss not in ld:
                env["LD_LIBRARY_PATH"] = f"{cudss}:{ld}" if ld else cudss
            info(f"re-exec into {py} (needs '{mod}' for {why})")
            os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], env)
    fail(f"module '{mod}' not found in any known interpreter (needed for {why})")
    print(f"       searched: {', '.join(CANDIDATE_PYTHONS)}")
    sys.exit(3)


# ---------------------------------------------------------------------------
# camera discovery
# ---------------------------------------------------------------------------

KNOWN_VENDORS = {
    "8086": "Intel (RealSense)",
    "2bc5": "Orbbec",
    "0c45": "Microdia / Sonix (Innomaker OEM)",
    "046d": "Logitech",
    "05a3": "ARC / generic UVC",
    "1e4e": "Etron / Cubeternet",
    "32e4": "Arducam USB",
    "0bda": "Realtek",
}

REALSENSE_MODELS = {
    "0b3a": "RealSense Depth Camera D435i",
    "0b07": "RealSense Depth Camera D435",
    "0b64": "RealSense LiDAR Camera L515",
    "0b5c": "RealSense Depth Camera D455",
    "0ad3": "RealSense D415",
    "0b5b": "RealSense D405",
    "0b3d": "RealSense D455",
}


@dataclass
class VideoNode:
    """One /dev/videoN node."""
    dev: str
    index: int
    name: str = ""
    driver: str = ""
    bus_info: str = ""
    caps: list[str] = field(default_factory=list)
    is_capture: bool = False
    is_metadata: bool = False
    formats: list[dict] = field(default_factory=list)
    udev: dict = field(default_factory=dict)


@dataclass
class Camera:
    """A logical camera, possibly spanning several /dev/video nodes."""
    kind: str                      # csi | realsense | orbbec | uvc | unknown
    label: str
    spec: str                      # canonical selector, e.g. csi:0 / /dev/video8
    nodes: list[VideoNode] = field(default_factory=list)
    vendor_id: str = ""
    product_id: str = ""
    serial: str = ""
    bus: str = ""
    usb_speed: str = ""
    sensor_id: Optional[int] = None      # CSI Argus sensor-id
    color_node: Optional[str] = None
    depth_node: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nodes"] = [{"dev": n.dev, "name": n.name, "driver": n.driver,
                       "metadata": n.is_metadata,
                       "formats": [f["fourcc"] for f in n.formats]} for n in self.nodes]
        return d


def _parse_v4l2_all(dev: str) -> dict:
    rc, out, _ = run(["v4l2-ctl", "-d", dev, "--info"], timeout=10)
    if rc != 0:
        return {}
    res: dict = {"caps": [], "device_caps": []}
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Driver name"):
            res["driver"] = s.split(":", 1)[1].strip()
        elif s.startswith("Card type"):
            res["name"] = s.split(":", 1)[1].strip()
        elif s.startswith("Bus info"):
            res["bus_info"] = s.split(":", 1)[1].strip()
        elif s.startswith("Device Caps"):
            section = "device_caps"
        elif s.startswith("Capabilities"):
            section = "caps"
        elif section and s and not s.endswith(":") and ":" not in s:
            res[section].append(s)
    return res


def _parse_formats(dev: str) -> list[dict]:
    rc, out, _ = run(["v4l2-ctl", "-d", dev, "--list-formats-ext"], timeout=10)
    if rc != 0:
        return []
    formats: list[dict] = []
    cur: Optional[dict] = None
    size: Optional[str] = None
    for line in out.splitlines():
        s = line.strip()
        # fourcc can contain trailing spaces, e.g. 'Z16 '
        m = re.match(r"\[\d+\]:\s+'(.{1,4}?)\s*'\s*\((.*?)\)", s)
        if m:
            cur = {"fourcc": m.group(1), "desc": m.group(2), "sizes": {}}
            formats.append(cur)
            continue
        m = re.match(r"Size:\s+\S+\s+(\d+x\d+)", s)
        if m and cur is not None:
            size = m.group(1)
            cur["sizes"].setdefault(size, [])
            continue
        m = re.match(r"Interval:\s+\S+\s+[\d.]+s\s+\(([\d.]+)\s*fps\)", s)
        if m and cur is not None and size:
            fps = float(m.group(1))
            if fps not in cur["sizes"][size]:
                cur["sizes"][size].append(fps)
    return formats


def _udev_props(dev: str) -> dict:
    rc, out, _ = run(["udevadm", "info", "-q", "property", "-n", dev], timeout=10)
    if rc != 0:
        return {}
    props = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v
    return props


def _usb_speed_from_devpath(devpath: str) -> str:
    """Walk up sysfs from a video node DEVPATH to the USB device dir and read speed."""
    if not devpath:
        return ""
    p = "/sys" + devpath
    for _ in range(8):
        p = os.path.dirname(p)
        if not p.startswith("/sys/devices"):
            break
        speed = read_file(os.path.join(p, "speed"))
        if speed:
            mapping = {"1.5": "USB1.0 low (1.5M)", "12": "USB1.1 (12M)",
                       "480": "USB2.0 (480M)", "5000": "USB3.0 (5G)",
                       "10000": "USB3.1 (10G)", "20000": "USB3.2 (20G)"}
            return mapping.get(speed, f"{speed}M")
    return ""


def enumerate_video_nodes(with_formats: bool = True) -> list[VideoNode]:
    nodes: list[VideoNode] = []
    devs = sorted(glob.glob("/dev/video*"),
                  key=lambda d: int(re.sub(r"\D", "", d) or 0))
    for dev in devs:
        idx = int(re.sub(r"\D", "", dev) or 0)
        n = VideoNode(dev=dev, index=idx)
        n.name = read_file(f"/sys/class/video4linux/video{idx}/name")
        if have("v4l2-ctl"):
            meta = _parse_v4l2_all(dev)
            n.driver = meta.get("driver", "")
            n.name = meta.get("name", n.name)
            n.bus_info = meta.get("bus_info", "")
            dcaps = meta.get("device_caps", []) or meta.get("caps", [])
            n.caps = dcaps
            joined = " ".join(dcaps).lower()
            n.is_metadata = "metadata capture" in joined
            n.is_capture = "video capture" in joined
        n.udev = _udev_props(dev)
        if with_formats and n.is_capture:
            n.formats = _parse_formats(dev)
        nodes.append(n)
    return nodes


def _csi_sensor_map(csi_nodes: list[VideoNode]) -> dict[str, int]:
    """
    Map CSI /dev/videoN -> Argus sensor-id.

    Argus enumerates sensors in device-tree order, which on this board matches
    ascending i2c bus number of the imx219 sensors (e.g. 9-0010 -> 0, 10-0010 -> 1).
    """
    parsed = []
    for n in csi_nodes:
        m = re.search(r"(\d+)-([0-9a-f]{4})", n.name)
        bus = int(m.group(1)) if m else n.index
        parsed.append((bus, n.dev))
    parsed.sort()
    return {dev: i for i, (_bus, dev) in enumerate(parsed)}


def discover_cameras(with_formats: bool = True) -> list[Camera]:
    nodes = enumerate_video_nodes(with_formats=with_formats)
    cams: list[Camera] = []

    csi_nodes = [n for n in nodes if "tegra" in n.driver.lower() or "tegra" in n.bus_info.lower()]
    csi_nodes = [n for n in csi_nodes if n.is_capture]
    sensor_map = _csi_sensor_map(csi_nodes)
    for n in csi_nodes:
        sid = sensor_map.get(n.dev, 0)
        sensor = "unknown"
        m = re.search(r"(imx\d+|ov\d+)", n.name, re.I)
        if m:
            sensor = m.group(1).upper()
        cam = Camera(
            kind="csi",
            label=f"CSI camera {sid} ({sensor})",
            spec=f"csi:{sid}",
            nodes=[n],
            sensor_id=sid,
            bus=n.bus_info,
            color_node=n.dev,
        )
        cam.extra["sensor"] = sensor
        cam.extra["note"] = ("raw Bayer over V4L2; use Argus/nvarguscamerasrc, "
                             "not plain OpenCV VideoCapture")
        cams.append(cam)

    # group remaining USB nodes by udev ID_PATH prefix (one physical device)
    usb_nodes = [n for n in nodes if n not in csi_nodes and n.udev.get("ID_VENDOR_ID")]
    groups: dict[str, list[VideoNode]] = {}
    for n in usb_nodes:
        path = n.udev.get("ID_PATH", n.bus_info)
        key = re.sub(r":\d+\.\d+$", "", path) + "|" + n.udev.get("ID_SERIAL", "")
        groups.setdefault(key, []).append(n)

    for key, gnodes in groups.items():
        gnodes.sort(key=lambda x: x.index)
        first = gnodes[0]
        vid = first.udev.get("ID_VENDOR_ID", "")
        pid = first.udev.get("ID_MODEL_ID", "")
        serial = first.udev.get("ID_SERIAL_SHORT", "")
        capture_nodes = [n for n in gnodes if n.is_capture and not n.is_metadata]
        speed = _usb_speed_from_devpath(first.udev.get("DEVPATH", ""))

        if vid == "8086":
            model = REALSENSE_MODELS.get(pid, f"RealSense (pid {pid})")
            color = None
            depth = None
            for n in capture_nodes:
                fcc = [f["fourcc"] for f in n.formats]
                if "YUYV" in fcc or "MJPG" in fcc:
                    color = color or n.dev
                if "Z16" in fcc or "Z16 " in fcc:
                    depth = depth or n.dev
            cam = Camera(kind="realsense", label=model, spec=f"rs:{serial or pid}",
                         nodes=gnodes, vendor_id=vid, product_id=pid, serial=serial,
                         bus=first.bus_info, usb_speed=speed,
                         color_node=color, depth_node=depth)
            cam.extra["note"] = ("SDK streams available as rs:color / rs:depth / rs:ir "
                                 "(needs pyrealsense2)")
        elif vid == "2bc5":
            cam = Camera(kind="orbbec", label=f"Orbbec camera (pid {pid})",
                         spec=f"/dev/video{capture_nodes[0].index}" if capture_nodes else "orbbec",
                         nodes=gnodes, vendor_id=vid, product_id=pid, serial=serial,
                         bus=first.bus_info, usb_speed=speed,
                         color_node=capture_nodes[0].dev if capture_nodes else None)
            cam.extra["note"] = "full depth access needs pyorbbecsdk2"
        else:
            label = first.udev.get("ID_V4L_PRODUCT") or first.name or "USB camera"
            cam = Camera(kind="uvc", label=f"USB UVC camera: {label}",
                         spec=capture_nodes[0].dev if capture_nodes else first.dev,
                         nodes=gnodes, vendor_id=vid, product_id=pid, serial=serial,
                         bus=first.bus_info, usb_speed=speed,
                         color_node=capture_nodes[0].dev if capture_nodes else None)
        cams.append(cam)

    # anything left over (non-USB, non-CSI capture nodes)
    claimed = {n.dev for cam in cams for n in cam.nodes}
    for n in nodes:
        if n.dev not in claimed and n.is_capture and not n.is_metadata:
            cams.append(Camera(kind="unknown", label=n.name or "unknown capture device",
                               spec=n.dev, nodes=[n], color_node=n.dev))
    return cams


def realsense_sdk_devices() -> list[dict]:
    """Query the RealSense SDK. Returns [] if pyrealsense2 is unavailable."""
    try:
        import pyrealsense2 as rs  # noqa
    except ImportError:
        return []
    import pyrealsense2 as rs
    out = []
    try:
        ctx = rs.context()
        for d in ctx.query_devices():
            def gi(key, default="?"):
                try:
                    return d.get_info(getattr(rs.camera_info, key))
                except Exception:
                    return default
            entry = {
                "name": gi("name"), "serial": gi("serial_number"),
                "firmware": gi("firmware_version"), "usb": gi("usb_type_descriptor"),
                "product_line": gi("product_line"), "streams": [],
            }
            try:
                for s in d.sensors:
                    sname = s.get_info(rs.camera_info.name)
                    profs = set()
                    for p in s.get_stream_profiles():
                        try:
                            vp = p.as_video_stream_profile()
                            profs.add((str(p.stream_type()).split(".")[-1],
                                       vp.width(), vp.height(), p.fps(),
                                       str(p.format()).split(".")[-1]))
                        except Exception:
                            continue
                    top = sorted(profs, key=lambda x: (-x[1] * x[2], -x[3]))[:6]
                    entry["streams"].append({"sensor": sname, "top_profiles": top})
            except Exception:
                pass
            out.append(entry)
    except Exception as e:
        return [{"error": str(e)}]
    return out


def orbbec_sdk_devices() -> list[dict]:
    try:
        import pyorbbecsdk2 as ob  # type: ignore
    except ImportError:
        return []
    try:
        ctx = ob.Context()  # type: ignore
        devs = ctx.query_devices()
        out = []
        for i in range(len(devs)):
            d = devs[i]
            di = d.get_device_info()
            out.append({"name": di.name(), "serial": di.serial_number(),
                        "pid": hex(di.pid()), "firmware": di.firmware_version()})
        return out
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# camera list / formats
# ---------------------------------------------------------------------------

def cmd_camera_list(args) -> int:
    cams = discover_cameras(with_formats=True)
    rs_devs = realsense_sdk_devices()
    ob_devs = orbbec_sdk_devices()

    if args.json:
        print(json.dumps({
            "cameras": [c_.to_dict() for c_ in cams],
            "realsense_sdk": rs_devs,
            "orbbec_sdk": ob_devs,
        }, indent=2))
        return 0

    hdr(f"Cameras detected: {len(cams)}")
    if not cams:
        warn("no /dev/video* capture devices found")
    kind_color = {"csi": "magenta", "realsense": "blue", "orbbec": "cyan",
                  "uvc": "green", "unknown": "grey"}
    for i, cam in enumerate(cams):
        print(f"\n[{i}] {c(cam.label, kind_color.get(cam.kind, 'bold'))}")
        print(f"      kind      : {cam.kind}")
        print(f"      selector  : {c(cam.spec, 'yellow')}")
        if cam.vendor_id:
            vend = KNOWN_VENDORS.get(cam.vendor_id, "")
            print(f"      usb id    : {cam.vendor_id}:{cam.product_id}"
                  + (f"  ({vend})" if vend else ""))
        if cam.serial:
            print(f"      serial    : {cam.serial}")
        if cam.usb_speed:
            print(f"      link      : {cam.usb_speed}")
        if cam.bus:
            print(f"      bus       : {cam.bus}")
        if cam.sensor_id is not None:
            print(f"      argus id  : sensor-id={cam.sensor_id}")
        if cam.color_node:
            print(f"      color node: {cam.color_node}")
        if cam.depth_node:
            print(f"      depth node: {cam.depth_node}")
        nodelist = ", ".join(
            f"{n.dev}{'(meta)' if n.is_metadata else ''}" for n in cam.nodes)
        print(f"      v4l2 nodes: {nodelist}")
        if cam.extra.get("note"):
            print(f"      note      : {c(cam.extra['note'], 'grey')}")
        if args.formats:
            for n in cam.nodes:
                if n.is_metadata or not n.formats:
                    continue
                print(f"      formats {n.dev}:")
                for f in n.formats:
                    sizes = []
                    for size, fps in sorted(
                            f["sizes"].items(),
                            key=lambda kv: -int(kv[0].split("x")[0]) * int(kv[0].split("x")[1])):
                        fpss = "/".join(str(int(x)) for x in sorted(set(fps), reverse=True))
                        sizes.append(f"{size}@{fpss}" if fpss else size)
                    shown = sizes if args.all_modes else sizes[:6]
                    more = "" if len(shown) == len(sizes) else f"  (+{len(sizes)-len(shown)} more)"
                    print(f"        {f['fourcc']:<6} {f['desc'][:28]:<28} "
                          f"{', '.join(shown)}{more}")

    if rs_devs:
        hdr("RealSense SDK (pyrealsense2)")
        for d in rs_devs:
            if "error" in d:
                fail(f"SDK error: {d['error']}")
                continue
            ok(f"{d['name']}  serial={d['serial']}  fw={d['firmware']}  usb={d['usb']}")
            for s in d["streams"]:
                print(f"        sensor: {s['sensor']}")
                for p in s["top_profiles"]:
                    print(f"          {p[0]:<12} {p[1]}x{p[2]}@{p[3]} {p[4]}")
    else:
        rs_present = any(c_.kind == "realsense" for c_ in cams)
        if rs_present:
            hdr("RealSense SDK (pyrealsense2)")
            warn("pyrealsense2 not importable in this interpreter; "
                 "depth/IR streams unavailable here")
            info("try: ~/lerobot-py310-cuda/bin/python jetson_devices.py camera list")

    if ob_devs:
        hdr("Orbbec SDK (pyorbbecsdk2)")
        for d in ob_devs:
            if "error" in d:
                fail(f"SDK error: {d['error']}")
            else:
                ok(f"{d['name']}  serial={d['serial']}  pid={d['pid']}  fw={d['firmware']}")
    elif any(c_.kind == "orbbec" for c_ in cams):
        hdr("Orbbec SDK (pyorbbecsdk2)")
        warn("Orbbec USB device present but pyorbbecsdk2 is not installed")
        info("install: ~/.local/bin/uv pip install --python <env>/bin/python pyorbbecsdk2")

    hdr("Selectors you can pass to --camera")
    for i, cam in enumerate(cams):
        print(f"  {i:<3} {cam.spec:<16} {cam.label}")
    if rs_devs:
        print(f"  {'':<3} {'rs:color':<16} RealSense SDK color stream")
        print(f"  {'':<3} {'rs:depth':<16} RealSense SDK depth (colormapped)")
        print(f"  {'':<3} {'rs:ir':<16} RealSense SDK left infrared")
        print(f"  {'':<3} {'rs:rgbd':<16} RealSense SDK color+depth side by side")
    return 0


def cmd_camera_formats(args) -> int:
    dev = args.device
    if not dev.startswith("/dev/"):
        cam = resolve_camera(dev)
        dev = cam.color_node or cam.nodes[0].dev
    hdr(f"Formats for {dev}")
    rc, out, err = run(["v4l2-ctl", "-d", dev, "--list-formats-ext"], timeout=15)
    print(out or err)
    return rc


# ---------------------------------------------------------------------------
# camera source resolution and GStreamer pipelines
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """A resolved capture source with everything needed to build pipelines."""
    kind: str              # csi | v4l2 | rs_sdk
    label: str
    camera: Optional[Camera] = None
    dev: Optional[str] = None
    sensor_id: Optional[int] = None
    rs_stream: str = "color"     # color | depth | ir | rgbd
    rs_serial: Optional[str] = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: Optional[str] = None
    flip: Optional[int] = None


def _pick_mode(node: VideoNode, want_w: Optional[int], want_h: Optional[int],
               want_fps: Optional[int]) -> tuple[str, int, int, int]:
    """Choose (fourcc, w, h, fps) for a V4L2 node, preferring MJPG for bandwidth."""
    prefer = ["MJPG", "YUYV", "UYVY", "NV12", "GREY", "Z16", "Z16 "]
    fmts = sorted(node.formats,
                  key=lambda f: prefer.index(f["fourcc"]) if f["fourcc"] in prefer else 99)
    for f in fmts:
        candidates = []
        for size, fpss in f["sizes"].items():
            w, h = (int(x) for x in size.split("x"))
            for fp in (fpss or [30.0]):
                candidates.append((w, h, int(fp)))
        if not candidates:
            continue
        if want_w and want_h:
            match = [x for x in candidates if x[0] == want_w and x[1] == want_h]
            if want_fps:
                exact = [x for x in match if x[2] == want_fps]
                match = exact or match
            if match:
                m = max(match, key=lambda x: x[2])
                return f["fourcc"], m[0], m[1], m[2]
            continue
        # no explicit size: prefer 1280x720, else largest at >=30fps, else largest
        pref = [x for x in candidates if (x[0], x[1]) == (1280, 720)]
        if pref:
            m = max(pref, key=lambda x: x[2])
            return f["fourcc"], m[0], m[1], m[2]
        fast = [x for x in candidates if x[2] >= 30]
        pool = fast or candidates
        m = max(pool, key=lambda x: (x[0] * x[1], x[2]))
        return f["fourcc"], m[0], m[1], m[2]
    return ("YUYV", want_w or 640, want_h or 480, want_fps or 30)


def resolve_camera(spec: str, cams: Optional[list[Camera]] = None) -> Camera:
    cams = cams if cams is not None else discover_cameras(with_formats=True)
    if not cams:
        raise SystemExit("no cameras detected")
    s = spec.strip()
    if s.isdigit() and int(s) < len(cams):
        return cams[int(s)]
    for cam in cams:
        if cam.spec == s:
            return cam
    if s.startswith("/dev/video"):
        for cam in cams:
            if any(n.dev == s for n in cam.nodes):
                return cam
    low = s.lower()
    alias = {"usb": "uvc", "uvc": "uvc", "csi": "csi", "realsense": "realsense",
             "rs": "realsense", "orbbec": "orbbec"}
    if low in alias:
        for cam in cams:
            if cam.kind == alias[low]:
                return cam
    for cam in cams:
        if low in cam.label.lower() or low in cam.spec.lower():
            return cam
    raise SystemExit(f"camera '{spec}' not found; run: jetson_devices.py camera list")


def resolve_source(spec: str, args) -> Source:
    """Turn a --camera selector into a Source."""
    s = spec.strip()
    low = s.lower()
    if low.startswith("rs:") and low.split(":")[-1] in ("color", "depth", "ir", "rgbd"):
        parts = s.split(":")
        stream = parts[-1].lower()
        serial = parts[1] if len(parts) == 3 else None
        return Source(kind="rs_sdk", label=f"RealSense SDK {stream}",
                      rs_stream=stream, rs_serial=serial,
                      width=args.width or 640, height=args.height or 480,
                      fps=args.fps or 30)

    cam = resolve_camera(s)
    if cam.kind == "csi":
        return Source(kind="csi", label=cam.label, camera=cam, sensor_id=cam.sensor_id,
                      width=args.width or 1280, height=args.height or 720,
                      fps=args.fps or 30,
                      flip=args.flip if args.flip is not None else 2)
    dev = cam.color_node or (cam.nodes[0].dev if cam.nodes else None)
    if not dev:
        raise SystemExit(f"camera '{spec}' has no usable capture node")
    node = next((n for n in cam.nodes if n.dev == dev), cam.nodes[0])
    fourcc, w, h, fps = _pick_mode(node, args.width, args.height, args.fps)
    return Source(kind="v4l2", label=f"{cam.label} [{dev}]", camera=cam, dev=dev,
                  width=w, height=h, fps=fps, fourcc=fourcc,
                  flip=args.flip)


def gst_env() -> dict:
    """Argus over SSH fails with FrameConsumer errors when DISPLAY is set."""
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    return env


def _flip_element(flip: Optional[int]) -> str:
    return f"nvvidconv flip-method={flip}" if flip is not None else "nvvidconv"


def _v4l2_flip(flip: Optional[int]) -> str:
    # videoflip method: 2 = 180 rotation (matches nvvidconv flip-method=2 intent)
    mapping = {0: "none", 1: "counterclockwise", 2: "rotate-180",
               3: "clockwise", 4: "horizontal-flip", 5: "vertical-flip"}
    if flip is None or flip == 0:
        return ""
    return f"videoflip method={mapping.get(flip, 'rotate-180')} ! "


def pipeline_jpeg(src: Source, num_buffers: Optional[int] = None) -> Optional[list[str]]:
    """A gst-launch argv producing a raw stream of JPEGs on stdout."""
    nb = f"num-buffers={num_buffers} " if num_buffers else ""
    if src.kind == "csi":
        p = (f"nvarguscamerasrc sensor-id={src.sensor_id} {nb}"
             f"tnr-mode=2 ee-mode=2 aeantibanding=3 ! "
             f"video/x-raw(memory:NVMM),width={src.width},height={src.height},"
             f"framerate={src.fps}/1 ! "
             f"{_flip_element(src.flip)} ! video/x-raw,format=I420 ! "
             f"jpegenc quality=85 ! fdsink fd=1")
    elif src.kind == "v4l2":
        if src.fourcc == "MJPG":
            p = (f"v4l2src device={src.dev} {nb}! "
                 f"image/jpeg,width={src.width},height={src.height},"
                 f"framerate={src.fps}/1 ! fdsink fd=1")
        else:
            fmt = {"YUYV": "YUY2", "UYVY": "UYVY", "NV12": "NV12",
                   "GREY": "GRAY8", "Z16": "GRAY16_LE", "Z16 ": "GRAY16_LE"}.get(
                       src.fourcc or "YUYV", "YUY2")
            p = (f"v4l2src device={src.dev} {nb}! "
                 f"video/x-raw,format={fmt},width={src.width},height={src.height},"
                 f"framerate={src.fps}/1 ! "
                 f"{_v4l2_flip(src.flip)}videoconvert ! jpegenc quality=85 ! fdsink fd=1")
    else:
        return None
    return ["gst-launch-1.0", "-q"] + p.split()


def pipeline_record(src: Source, outfile: str, seconds: float) -> Optional[list[str]]:
    frames = max(1, int(seconds * src.fps))
    if src.kind == "csi":
        p = (f"nvarguscamerasrc sensor-id={src.sensor_id} num-buffers={frames} "
             f"tnr-mode=2 ee-mode=2 aeantibanding=3 ! "
             f"video/x-raw(memory:NVMM),width={src.width},height={src.height},"
             f"framerate={src.fps}/1 ! "
             f"{_flip_element(src.flip)} ! video/x-raw,format=I420 ! "
             f"x264enc tune=zerolatency speed-preset=ultrafast key-int-max={src.fps} ! "
             f"h264parse ! mp4mux ! filesink location={outfile}")
    elif src.kind == "v4l2":
        if src.fourcc == "MJPG":
            head = (f"v4l2src device={src.dev} num-buffers={frames} ! "
                    f"image/jpeg,width={src.width},height={src.height},"
                    f"framerate={src.fps}/1 ! jpegparse ! jpegdec ! ")
        else:
            fmt = {"YUYV": "YUY2", "UYVY": "UYVY", "NV12": "NV12",
                   "GREY": "GRAY8"}.get(src.fourcc or "YUYV", "YUY2")
            head = (f"v4l2src device={src.dev} num-buffers={frames} ! "
                    f"video/x-raw,format={fmt},width={src.width},height={src.height},"
                    f"framerate={src.fps}/1 ! ")
        p = (head + _v4l2_flip(src.flip) +
             f"videoconvert ! "
             f"x264enc tune=zerolatency speed-preset=ultrafast key-int-max={src.fps} ! "
             f"h264parse ! mp4mux ! filesink location={outfile}")
    else:
        return None
    return ["gst-launch-1.0", "-e", "-q"] + p.split()


def pick_videosink(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        return ""
    # X11-forwarded display cannot use the EGL sink
    if disp.startswith("localhost:") or disp.startswith("127.0.0.1:"):
        return "ximagesink sync=false"
    return "nveglglessink sync=false"


def pipeline_window(src: Source, sink: str) -> Optional[list[str]]:
    if src.kind == "csi":
        p = (f"nvarguscamerasrc sensor-id={src.sensor_id} "
             f"tnr-mode=2 ee-mode=2 aeantibanding=3 ! "
             f"video/x-raw(memory:NVMM),width={src.width},height={src.height},"
             f"framerate={src.fps}/1 ! "
             f"{_flip_element(src.flip)} ! video/x-raw,format=I420 ! "
             f"videoconvert ! {sink}")
    elif src.kind == "v4l2":
        if src.fourcc == "MJPG":
            head = (f"v4l2src device={src.dev} ! image/jpeg,width={src.width},"
                    f"height={src.height},framerate={src.fps}/1 ! jpegparse ! jpegdec ! ")
        else:
            fmt = {"YUYV": "YUY2", "UYVY": "UYVY", "NV12": "NV12",
                   "GREY": "GRAY8"}.get(src.fourcc or "YUYV", "YUY2")
            head = (f"v4l2src device={src.dev} ! video/x-raw,format={fmt},"
                    f"width={src.width},height={src.height},framerate={src.fps}/1 ! ")
        p = head + _v4l2_flip(src.flip) + f"videoconvert ! {sink}"
    else:
        return None
    return ["gst-launch-1.0", "-q"] + p.split()


# ---------------------------------------------------------------------------
# JPEG frame producers
# ---------------------------------------------------------------------------

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def gst_jpeg_frames(argv: list[str], stop: threading.Event,
                    on_stderr: Optional[Callable[[str], None]] = None) -> Iterator[bytes]:
    """Run a gst-launch pipeline and yield complete JPEG frames from its stdout."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=gst_env(), bufsize=0)
    err_lines: list[str] = []

    def drain():
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            err_lines.append(line)
            if on_stderr:
                on_stderr(line)
    t = threading.Thread(target=drain, daemon=True)
    t.start()

    buf = bytearray()
    try:
        assert proc.stdout is not None
        while not stop.is_set():
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                start = buf.find(SOI)
                if start < 0:
                    if len(buf) > 4 << 20:
                        del buf[:-2]
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    if start > 0:
                        del buf[:start]
                    break
                frame = bytes(buf[start:end + 2])
                del buf[:end + 2]
                yield frame
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not stop.is_set() and proc.returncode not in (0, None, -2):
            tail = "\n        ".join(err_lines[-6:])
            if tail:
                sys.stderr.write(f"  gstreamer stderr:\n        {tail}\n")


def rs_sdk_frames(src: Source, stop: threading.Event) -> Iterator[bytes]:
    """Yield JPEG frames from the RealSense SDK (color / depth / ir / rgbd)."""
    rs = require_module("pyrealsense2", "RealSense SDK streams")
    cv2 = require_module("cv2", "JPEG encoding of SDK frames")
    np = require_module("numpy", "SDK frame handling")

    pipe = rs.pipeline()
    cfg = rs.config()
    if src.rs_serial:
        cfg.enable_device(src.rs_serial)
    want = src.rs_stream
    if want in ("color", "rgbd"):
        cfg.enable_stream(rs.stream.color, src.width, src.height, rs.format.bgr8, src.fps)
    if want in ("depth", "rgbd"):
        cfg.enable_stream(rs.stream.depth, src.width, src.height, rs.format.z16, src.fps)
    if want == "ir":
        cfg.enable_stream(rs.stream.infrared, 1, src.width, src.height, rs.format.y8, src.fps)
    profile = pipe.start(cfg)
    try:
        dev = profile.get_device()
        sys.stderr.write(f"  RealSense: {dev.get_info(rs.camera_info.name)} "
                         f"serial {dev.get_info(rs.camera_info.serial_number)}\n")
        while not stop.is_set():
            frames = pipe.wait_for_frames(5000)
            imgs = []
            if want in ("color", "rgbd"):
                cf = frames.get_color_frame()
                if cf:
                    imgs.append(np.asanyarray(cf.get_data()))
            if want in ("depth", "rgbd"):
                df = frames.get_depth_frame()
                if df:
                    d = np.asanyarray(df.get_data())
                    dc = cv2.applyColorMap(
                        cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)
                    imgs.append(dc)
            if want == "ir":
                irf = frames.get_infrared_frame(1)
                if irf:
                    g = np.asanyarray(irf.get_data())
                    imgs.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            if not imgs:
                continue
            img = imgs[0] if len(imgs) == 1 else np.hstack(
                [cv2.resize(x, (imgs[0].shape[1], imgs[0].shape[0])) for x in imgs])
            okj, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if okj:
                yield enc.tobytes()
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


def frames_for(src: Source, stop: threading.Event,
               num_buffers: Optional[int] = None) -> Iterator[bytes]:
    if src.kind == "rs_sdk":
        return rs_sdk_frames(src, stop)
    argv = pipeline_jpeg(src, num_buffers=num_buffers)
    if argv is None:
        raise SystemExit(f"no capture pipeline for source kind '{src.kind}'")
    return gst_jpeg_frames(argv, stop)


# ---------------------------------------------------------------------------
# camera snap / record / show / test
# ---------------------------------------------------------------------------

def cmd_camera_snap(args) -> int:
    src = resolve_source(args.camera, args)
    outdir = args.outdir or DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)
    count = max(1, args.count)
    hdr(f"Snapshot: {src.label}")
    info(f"{src.width}x{src.height}@{src.fps}"
         + (f" {src.fourcc}" if src.fourcc else "")
         + (f" flip={src.flip}" if src.flip else ""))
    stop = threading.Event()
    warmup = args.warmup
    if warmup == 3 and src.kind == "rs_sdk":
        # RealSense RGB auto-exposure needs more frames to converge
        warmup = 15
    saved: list[str] = []
    t0 = time.time()
    try:
        n = 0
        for jpg in frames_for(src, stop,
                              num_buffers=(warmup + count) if src.kind != "rs_sdk" else None):
            n += 1
            if n <= warmup:
                continue
            base = args.out if args.out and count == 1 else None
            if base is None:
                tag = re.sub(r"[^A-Za-z0-9]+", "_", src.label).strip("_").lower()[:40]
                base = os.path.join(outdir, f"{tag}_{timestamp()}_{len(saved):02d}.jpg")
            elif not os.path.isabs(base):
                base = os.path.join(outdir, base)
            with open(base, "wb") as f:
                f.write(jpg)
            saved.append(base)
            ok(f"saved {base}  ({len(jpg)/1024:.1f} KB)")
            if len(saved) >= count:
                stop.set()
                break
    except KeyboardInterrupt:
        stop.set()
    if not saved:
        fail(f"no frames captured after {time.time()-t0:.1f}s")
        return 1
    return 0


def cmd_camera_record(args) -> int:
    src = resolve_source(args.camera, args)
    outdir = args.outdir or DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)
    out = args.out
    if not out:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", src.label).strip("_").lower()[:40]
        out = os.path.join(outdir, f"{tag}_{src.width}x{src.height}_{src.fps}fps_"
                                   f"{timestamp()}.mp4")
    elif not os.path.isabs(out):
        out = os.path.join(outdir, out)

    hdr(f"Record: {src.label} -> {out}")
    info(f"{src.width}x{src.height}@{src.fps} for {args.duration}s "
         f"(software x264; Orin Nano has no NVENC)")

    argv = pipeline_record(src, out, args.duration)
    if argv is not None:
        t0 = time.time()
        proc = subprocess.Popen(argv, env=gst_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _, err = proc.communicate(timeout=args.duration + 60)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            _, err = proc.communicate(timeout=10)
        rc = proc.returncode
        if rc != 0:
            fail(f"gst-launch exited {rc}")
            print(err.decode("utf-8", "replace")[-1500:])
            return rc or 1
    else:
        # SDK source: JPEG frames -> gst mp4 encoder on stdin.
        # The Python encode path rarely sustains the sensor rate, so measure the
        # real delivery rate first and mux at it; otherwise playback runs fast.
        stop = threading.Event()
        gen = frames_for(src, stop)
        proc = None
        nframes = 0
        t0 = time.time()
        try:
            probe_n = 0
            probe_t0 = None
            for jpg in gen:
                probe_n += 1
                if probe_n == 3:            # skip warmup frames
                    probe_t0 = time.time()
                if probe_n >= 3 + 15:
                    break
            measured = (probe_n - 3) / max(time.time() - (probe_t0 or time.time()), 1e-3)
            mux_fps = max(1, min(src.fps, round(measured)))
            info(f"measured source rate {measured:.1f} fps -> muxing at {mux_fps} fps")

            enc = ["gst-launch-1.0", "-e", "-q", "fdsrc", "fd=0", "do-timestamp=true", "!",
                   "jpegparse", "!", "jpegdec", "!", "videoconvert", "!",
                   "videorate", "!", f"video/x-raw,framerate={mux_fps}/1", "!",
                   "x264enc", "tune=zerolatency", "speed-preset=ultrafast",
                   f"key-int-max={mux_fps}", "!", "h264parse", "!", "mp4mux",
                   "!", "filesink", f"location={out}"]
            proc = subprocess.Popen(enc, stdin=subprocess.PIPE, env=gst_env(),
                                    stderr=subprocess.PIPE)
            t0 = time.time()
            for jpg in gen:
                assert proc.stdin is not None
                proc.stdin.write(jpg)
                nframes += 1
                if time.time() - t0 >= args.duration:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            if proc is not None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
        info(f"{nframes} frames pushed to encoder "
             f"({nframes/max(time.time()-t0, 0.01):.1f} fps)")

    if os.path.exists(out) and os.path.getsize(out) > 1024:
        ok(f"{out}  ({os.path.getsize(out)/1e6:.2f} MB, {time.time()-t0:.1f}s wall)")
        return 0
    fail(f"output missing or empty: {out}")
    return 1


class MJPEGServer:
    """Minimal multi-client MJPEG-over-HTTP server fed by a JPEG frame iterator."""

    def __init__(self, port: int, title: str):
        self.port = port
        self.title = title
        self.frame: Optional[bytes] = None
        self.cond = threading.Condition()
        self.seq = 0
        self.stop = threading.Event()
        self._httpd = None

    def publish(self, jpg: bytes) -> None:
        with self.cond:
            self.frame = jpg
            self.seq += 1
            self.cond.notify_all()

    def serve_forever(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=jetsonframe")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    last = 0
                    try:
                        while not outer.stop.is_set():
                            with outer.cond:
                                if not outer.cond.wait_for(
                                        lambda: outer.seq != last and outer.frame, timeout=5):
                                    continue
                                jpg = outer.frame
                                last = outer.seq
                            self.wfile.write(b"--jetsonframe\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path.startswith("/snapshot"):
                    with outer.cond:
                        jpg = outer.frame
                    if not jpg:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                else:
                    body = (f"<!doctype html><html><head><title>{outer.title}</title>"
                            "<style>body{background:#111;color:#ddd;font-family:sans-serif;"
                            "text-align:center;margin:0;padding:12px}"
                            "img{max-width:100%;height:auto;border:1px solid #333}</style>"
                            f"</head><body><h3>{outer.title}</h3>"
                            "<img src='/stream'><p style='color:#888'>"
                            "/stream = MJPEG &nbsp;|&nbsp; /snapshot = single JPEG</p>"
                            "</body></html>").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def shutdown(self) -> None:
        self.stop.set()
        with self.cond:
            self.cond.notify_all()
        if self._httpd:
            self._httpd.shutdown()


def cmd_camera_show(args) -> int:
    src = resolve_source(args.camera, args)
    mode = args.mode
    if mode == "auto":
        mode = "window" if (os.environ.get("DISPLAY") and src.kind != "rs_sdk") else "stream"

    if mode == "window":
        sink = pick_videosink(args.sink)
        if not sink:
            warn("DISPLAY is not set; falling back to --mode stream")
            mode = "stream"
        else:
            argv = pipeline_window(src, sink)
            if argv is None:
                warn("no window pipeline for this source; falling back to --mode stream")
                mode = "stream"
            else:
                hdr(f"Window: {src.label}")
                info(f"sink={sink}  DISPLAY={os.environ.get('DISPLAY')}")
                info("close the window or press Ctrl-C to stop")
                env = os.environ.copy()   # window mode needs DISPLAY
                try:
                    return subprocess.call(argv, env=env)
                except KeyboardInterrupt:
                    return 0

    hdr(f"MJPEG stream: {src.label}")
    host = os.uname().nodename
    url = f"http://{host}:{args.port}/"
    server = MJPEGServer(args.port, f"{src.label} - {src.width}x{src.height}@{src.fps}")
    server.serve_forever()
    ok(f"open {c(url, 'yellow')}  (or http://<jetson-ip>:{args.port}/)")
    info("Ctrl-C to stop")
    stop = threading.Event()
    n = 0
    t0 = time.time()
    try:
        for jpg in frames_for(src, stop):
            server.publish(jpg)
            n += 1
            if n % 60 == 0:
                el = time.time() - t0
                print(f"\r  frames={n}  {n/el:5.1f} fps  ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        server.shutdown()
    print()
    ok(f"streamed {n} frames in {time.time()-t0:.1f}s")
    return 0 if n else 1


def _test_one(cam: Camera, args, results: list) -> None:
    class A:  # arg shim for resolve_source
        width = args.width
        height = args.height
        fps = args.fps
        flip = args.flip
    try:
        src = resolve_source(cam.spec, A)
    except SystemExit as e:
        results.append((cam.label, "SKIP", str(e)))
        return
    stop = threading.Event()
    t0 = time.time()
    got = 0
    sizes = []
    try:
        for jpg in frames_for(src, stop, num_buffers=args.frames + 3):
            got += 1
            sizes.append(len(jpg))
            if got >= args.frames:
                break
            if time.time() - t0 > args.timeout:
                break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        stop.set()
        results.append((cam.label, "FAIL", f"{type(e).__name__}: {e}"))
        return
    finally:
        stop.set()
    el = time.time() - t0
    if got >= 1:
        avg = sum(sizes) / len(sizes) / 1024
        results.append((cam.label, "PASS",
                        f"{got} frames in {el:.1f}s ({got/max(el,0.01):.1f} fps), "
                        f"{src.width}x{src.height}, avg {avg:.0f} KB/frame"))
    else:
        results.append((cam.label, "FAIL", f"no frames in {el:.1f}s"))


def cmd_camera_test(args) -> int:
    cams = discover_cameras(with_formats=True)
    if args.camera:
        cams = [resolve_camera(args.camera, cams)]
    hdr(f"Capture smoke test ({args.frames} frames each)")
    results: list[tuple[str, str, str]] = []
    for cam in cams:
        print(f"\n  testing {c(cam.label, 'bold')} ({cam.spec}) ...", flush=True)
        _test_one(cam, args, results)
        st = results[-1]
        (ok if st[1] == "PASS" else warn if st[1] == "SKIP" else fail)(f"{st[1]}: {st[2]}")

    # SDK-level RealSense checks
    if any(cm.kind == "realsense" for cm in cams) and not args.camera:
        for stream in ("color", "depth"):
            print(f"\n  testing {c(f'RealSense SDK {stream}', 'bold')} (rs:{stream}) ...",
                  flush=True)

            class A:
                width = args.width or 640
                height = args.height or 480
                fps = args.fps or 30
                flip = None
            try:
                src = resolve_source(f"rs:{stream}", A)
                stop = threading.Event()
                t0 = time.time()
                got = 0
                for _jpg in frames_for(src, stop):
                    got += 1
                    if got >= args.frames or time.time() - t0 > args.timeout:
                        break
                stop.set()
                if got:
                    results.append((f"RealSense SDK {stream}", "PASS",
                                    f"{got} frames in {time.time()-t0:.1f}s"))
                    ok(results[-1][2])
                else:
                    results.append((f"RealSense SDK {stream}", "FAIL", "no frames"))
                    fail("no frames")
            except SystemExit as e:
                results.append((f"RealSense SDK {stream}", "SKIP", str(e)))
                warn(str(e))
            except Exception as e:
                results.append((f"RealSense SDK {stream}", "FAIL", f"{type(e).__name__}: {e}"))
                fail(results[-1][2])

    hdr("Summary")
    npass = sum(1 for r in results if r[1] == "PASS")
    for label, status, detail in results:
        col = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}[status]
        print(f"  {c(status, col):<16} {label:<44} {detail}")
    print(f"\n  {npass}/{len(results)} passed")
    return 0 if npass == len(results) else 1


def cmd_camera_interactive(args) -> int:
    class A:
        width = None
        height = None
        fps = None
        flip = None
        out = None
        outdir = args.outdir
        count = 1
        warmup = 3
        duration = 5.0
        camera = ""
        mode = "auto"
        sink = None
        port = args.port
        frames = 10
        timeout = 20

    while True:
        cams = discover_cameras(with_formats=True)
        hdr("Camera menu")
        for i, cam in enumerate(cams):
            print(f"  {i}. {cam.label}  [{c(cam.spec, 'yellow')}]")
        base = len(cams)
        rs_specs = []
        if realsense_sdk_devices():
            for j, st in enumerate(("color", "depth", "ir", "rgbd")):
                print(f"  {base+j}. RealSense SDK {st}  [{c('rs:'+st, 'yellow')}]")
                rs_specs.append(f"rs:{st}")
        print(f"\n  r. rescan   t. test all   q. quit")
        try:
            sel = input(c("\n  select camera> ", "cyan")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if sel in ("q", "quit", "exit"):
            return 0
        if sel in ("r", ""):
            continue
        if sel == "t":
            A.camera = None
            cmd_camera_test(A)
            continue
        try:
            idx = int(sel)
        except ValueError:
            warn("enter a number, r, t or q")
            continue
        if idx < len(cams):
            spec = cams[idx].spec
            label = cams[idx].label
        elif idx - base < len(rs_specs):
            spec = rs_specs[idx - base]
            label = f"RealSense SDK {spec.split(':')[1]}"
        else:
            warn("out of range")
            continue

        while True:
            hdr(f"Selected: {label}  [{spec}]")
            print("  1. snapshot (jpg)")
            print("  2. record 5s mp4")
            print("  3. record N seconds")
            print("  4. live view - MJPEG in browser")
            print("  5. live view - X11 window")
            print("  6. quick capture test")
            print("  7. show v4l2 formats")
            print("  8. set resolution/fps")
            print("  b. back    q. quit")
            try:
                a = input(c("\n  action> ", "cyan")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            A.camera = spec
            if a == "b":
                break
            if a == "q":
                return 0
            try:
                if a == "1":
                    A.count, A.out = 1, None
                    cmd_camera_snap(A)
                elif a == "2":
                    A.duration, A.out = 5.0, None
                    cmd_camera_record(A)
                elif a == "3":
                    A.duration = float(input("  seconds> ").strip() or "5")
                    A.out = None
                    cmd_camera_record(A)
                elif a == "4":
                    A.mode = "stream"
                    cmd_camera_show(A)
                elif a == "5":
                    A.mode = "window"
                    cmd_camera_show(A)
                elif a == "6":
                    A.camera = spec if not spec.startswith("rs:") else None
                    A.frames = 10
                    cmd_camera_test(A)
                elif a == "7":
                    class F:
                        device = spec
                    if spec.startswith("rs:"):
                        warn("SDK stream: no single v4l2 node; use the parent RealSense entry")
                    else:
                        cmd_camera_formats(F)
                elif a == "8":
                    w = input("  width (blank=auto)> ").strip()
                    h = input("  height (blank=auto)> ").strip()
                    f = input("  fps (blank=auto)> ").strip()
                    fl = input("  flip 0-7 (blank=default)> ").strip()
                    A.width = int(w) if w else None
                    A.height = int(h) if h else None
                    A.fps = int(f) if f else None
                    A.flip = int(fl) if fl else None
                    ok(f"set {A.width}x{A.height}@{A.fps} flip={A.flip}")
                else:
                    warn("unknown action")
            except KeyboardInterrupt:
                print()
            except SystemExit as e:
                fail(str(e))
            except Exception as e:
                fail(f"{type(e).__name__}: {e}")
    return 0


# ---------------------------------------------------------------------------
# bluetooth
# ---------------------------------------------------------------------------

CONTROLLER_SIGNATURES = [
    # (regex on name, vid, pid, friendly, hint)
    (r"dualsense|wireless controller", "054c", "0ce6", "Sony DualSense (PS5)",
     "hold PS + Create until the light bar flashes blue"),
    (r"dualsense edge", "054c", "0df2", "Sony DualSense Edge (PS5)",
     "hold PS + Create until the light bar flashes"),
    (r"wireless controller", "054c", "09cc", "Sony DualShock 4 (PS4)",
     "hold PS + Share until the light bar flashes"),
    (r"quest|oculus|touch", "2833", "", "Meta Quest / Oculus device",
     "Quest Touch controllers normally bond to the headset, not to a host"),
    (r"xbox wireless controller", "045e", "", "Microsoft Xbox Wireless Controller",
     "hold the pair button on top until the Xbox button flashes fast"),
    (r"8bitdo", "2dc8", "", "8BitDo controller", ""),
    (r"nintendo|joy-con|pro controller", "057e", "", "Nintendo controller", ""),
]

PS5_NAME_RE = re.compile(r"dualsense|wireless controller", re.I)
QUEST_NAME_RE = re.compile(r"quest|oculus|touch\s*(plus|pro)?", re.I)


def classify_bt_name(name: str, vid: str = "", pid: str = "") -> tuple[str, str]:
    for rx, v, p, friendly, hint in CONTROLLER_SIGNATURES:
        if vid and v and vid.lower() == v and (not p or not pid or pid.lower() == p):
            return friendly, hint
        if name and re.search(rx, name, re.I):
            return friendly, hint
    return "", ""


# bluetoothctl colorises with ANSI wrapped in readline's \x01 / \x02 markers,
# so "[CHG]" arrives as "[\x01\x1b[0;93m\x02CHG\x01\x1b[0m\x02]". Strip both.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def bt(*cmdargs: str, timeout: int = 20) -> tuple[int, str, str]:
    rc, out, err = run(["bluetoothctl"] + list(cmdargs), timeout=timeout)
    return rc, strip_ansi(out), strip_ansi(err)


def bt_adapter_status() -> dict:
    st: dict = {}
    rc, out, _ = run(["systemctl", "is-active", "bluetooth"], timeout=10)
    st["service"] = out.strip() or "unknown"
    rc, out, _ = run(["rfkill", "list", "bluetooth"], timeout=10)
    st["rfkill"] = out.strip()
    st["soft_blocked"] = "Soft blocked: yes" in out
    st["hard_blocked"] = "Hard blocked: yes" in out
    rc, out, _ = bt("show", timeout=15)
    st["show"] = out.strip()
    m = re.search(r"Controller ([0-9A-F:]{17})", out)
    st["address"] = m.group(1) if m else ""
    for key in ("Name", "Powered", "Discoverable", "Pairable", "Discovering"):
        m = re.search(rf"^\s*{key}:\s*(.+)$", out, re.M)
        if m:
            st[key.lower()] = m.group(1).strip()
    rc, out, _ = run(["hciconfig", "-a"], timeout=10)
    st["hciconfig"] = out.strip()
    return st


def cmd_bt_status(args) -> int:
    st = bt_adapter_status()
    hdr("Bluetooth adapter")
    (ok if st["service"] == "active" else fail)(f"bluetooth.service: {st['service']}")
    if st["soft_blocked"] or st["hard_blocked"]:
        fail(f"rfkill blocked (soft={st['soft_blocked']} hard={st['hard_blocked']}) "
             f"-> sudo rfkill unblock bluetooth")
    else:
        ok("rfkill: unblocked")
    if st.get("address"):
        ok(f"controller {st['address']}  name={st.get('name','?')}  "
           f"powered={st.get('powered','?')}")
    else:
        fail("no Bluetooth controller reported by bluetoothctl")
    m = re.search(r"HCI Version: (.+)", st.get("hciconfig", ""))
    if m:
        info(f"HCI {m.group(1).strip()}")
    m = re.search(r"Manufacturer: (.+)", st.get("hciconfig", ""))
    if m:
        info(f"chip: {m.group(1).strip()}")

    hdr("Known (paired / trusted) devices")
    devs = bt_devices()
    if not devs:
        info("none")
    for d in devs:
        friendly, _ = classify_bt_name(d["name"])
        tags = []
        if d.get("connected"):
            tags.append(c("connected", "green"))
        if d.get("paired"):
            tags.append("paired")
        if d.get("trusted"):
            tags.append("trusted")
        print(f"  {d['mac']}  {d['name']:<28} {friendly or '':<32} {' '.join(tags)}")

    hdr("Input devices (/dev/input)")
    pads = list_input_devices()
    if not pads:
        info("no evdev gamepad-like device present")
    for p in pads:
        print(f"  {p['dev']:<20} {p['name']:<36} "
              f"{'readable' if p['readable'] else c('NOT READABLE', 'red')}")
    if pads and not all(p["readable"] for p in pads):
        warn(f"user '{os.environ.get('USER','?')}' cannot read some event nodes")
        info("logind normally grants a uaccess ACL to the active local seat user;")
        info("if there is no local session, run: jetson_devices.py bluetooth "
             "fix-permissions")
    return 0


def bt_devices(which: str = "") -> list[dict]:
    """List known devices; `which` may be 'Paired'/'Connected' (bluez 5.64 supports both)."""
    args_ = ["devices"] + ([which] if which else [])
    rc, out, _ = bt(*args_, timeout=15)
    devs = []
    for line in out.splitlines():
        m = re.match(r"Device ([0-9A-F:]{17})\s+(.*)", line.strip())
        if m:
            devs.append({"mac": m.group(1), "name": m.group(2).strip()})
    for d in devs:
        rc, out, _ = bt("info", d["mac"], timeout=15)
        d["paired"] = "Paired: yes" in out
        d["trusted"] = "Trusted: yes" in out
        d["connected"] = "Connected: yes" in out
        m = re.search(r"Name:\s*(.+)", out)
        if m:
            d["name"] = m.group(1).strip()
        m = re.search(r"Modalias:\s*usb:v([0-9A-Fa-f]{4})p([0-9A-Fa-f]{4})", out)
        if m:
            d["vid"], d["pid"] = m.group(1).lower(), m.group(2).lower()
        d["icon"] = (re.search(r"Icon:\s*(.+)", out).group(1).strip()
                     if re.search(r"Icon:\s*(.+)", out) else "")
    return devs


def is_connected(mac: str) -> bool:
    _rc, out, _err = bt("info", mac, timeout=15)
    return "Connected: yes" in out


def bt_scan(seconds: int, verbose: bool = True) -> list[dict]:
    """Run a timed scan, streaming discovered devices."""
    found: dict[str, dict] = {}
    for d in bt_devices():
        found[d["mac"]] = {"mac": d["mac"], "name": d["name"], "known": True,
                           "rssi": None}
    proc = subprocess.Popen(["bluetoothctl", "--timeout", str(seconds), "scan", "on"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    t0 = time.time()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = strip_ansi(line).strip()
            m = re.search(r"\[(NEW|CHG)\] Device ([0-9A-F:]{17})\s*(.*)", line)
            if not m:
                continue
            mac, rest = m.group(2), m.group(3).strip()
            e = found.setdefault(mac, {"mac": mac, "name": "", "known": False, "rssi": None})
            # [CHG] lines carry "Property: value"; only [NEW] and "Name:"/"Alias:"
            # updates actually name the device.
            prop = re.match(r"^([A-Za-z][A-Za-z0-9 ]*):\s*(.*)$", rest)
            if prop:
                key, val = prop.group(1).strip(), prop.group(2).strip()
                if key == "RSSI":
                    try:
                        e["rssi"] = int(val)
                    except ValueError:
                        pass
                elif key in ("Name", "Alias") and val:
                    e["name"] = val
            elif rest and not re.fullmatch(r"[0-9A-F-]{17}", rest):
                e["name"] = rest
            if verbose and m.group(1) == "NEW":
                friendly, _ = classify_bt_name(e["name"])
                tag = f"  <-- {c(friendly, 'green')}" if friendly else ""
                print(f"    +{time.time()-t0:5.1f}s  {mac}  {e['name'] or '(no name)'}{tag}")
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        run(["bluetoothctl", "scan", "off"], timeout=10)
    return sorted(found.values(), key=lambda d: (d["name"] == "", d["name"]))


def cmd_bt_scan(args) -> int:
    st = bt_adapter_status()
    if st["service"] != "active":
        fail("bluetooth.service is not active -> sudo systemctl start bluetooth")
        return 1
    if st.get("powered", "yes").lower() == "no":
        info("powering controller on")
        bt("power", "on")
    hdr(f"Scanning for {args.timeout}s")
    info("put the controller in pairing mode now:")
    print(f"      PS5 DualSense : hold {c('PS + Create', 'yellow')} "
          f"until the light bar flashes blue")
    print(f"      Meta Quest 3  : Touch Plus controllers pair to the headset; "
          f"a host only sees them if the headset released them")
    devs = bt_scan(args.timeout)
    hdr(f"Discovered {len(devs)} device(s)")
    interesting = []
    for d in devs:
        friendly, hint = classify_bt_name(d["name"])
        line = (f"  {d['mac']}  {(d['name'] or '(no name)'):<30} "
                f"{('RSSI ' + str(d['rssi'])) if d['rssi'] is not None else '':<10}"
                f"{'known' if d.get('known') else '':<7}")
        if friendly:
            interesting.append((d, friendly, hint))
            print(line + c(friendly, "green"))
        elif not args.all and not d["name"]:
            continue
        else:
            print(line)
    if interesting:
        hdr("Controllers found")
        for d, friendly, hint in interesting:
            ok(f"{friendly}  {d['mac']}")
            if hint:
                print(f"        {c(hint, 'grey')}")
            print(f"        connect: jetson_devices.py bluetooth pair {d['mac']}")
    else:
        warn("no known controller signature in range")
    return 0


# bluetoothctl's agent asks these at unpredictable moments. They must be answered
# immediately: a DualSense connects back inbound right after bonding and BlueZ
# denies the HID service if the prompt times out (~30s), killing the connection.
AGENT_PROMPTS = re.compile(
    r"(Confirm passkey.*\(yes/no\)"
    r"|Authorize service.*\(yes/no\)"
    r"|Accept pairing.*\(yes/no\)"
    r"|Request confirmation"
    r"|Request authorization)", re.I)


class BluetoothCtl:
    """Interactive bluetoothctl session driven over a pty (handles the agent prompts)."""

    def __init__(self, verbose: bool = True):
        import pty
        self.verbose = verbose
        self.answered = 0
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            ["bluetoothctl"], stdin=slave, stdout=slave, stderr=slave,
            close_fds=True, preexec_fn=os.setsid)
        os.close(slave)
        self.buf = ""
        time.sleep(0.5)
        self.read(0.5)

    def read(self, timeout: float = 1.0) -> str:
        import select
        out = ""
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.master], [], [], 0.2)
            if not r:
                continue
            try:
                data = os.read(self.master, 4096)
            except OSError:
                break
            if not data:
                break
            text = strip_ansi(data.decode("utf-8", "replace"))
            out += text
            if self.verbose:
                for line in text.splitlines():
                    s = line.strip()
                    if s and not s.startswith("[bluetooth]#"):
                        print(f"      {c(s, 'grey')}")
            for line in text.splitlines():
                if AGENT_PROMPTS.search(line):
                    self.answered += 1
                    print(f"      {c('-> auto-answering yes', 'green')}")
                    os.write(self.master, b"yes\n")
        self.buf += out
        return out

    def pump(self, seconds: float) -> str:
        """Keep reading (and auto-answering prompts) for a while."""
        acc = ""
        end = time.time() + seconds
        while time.time() < end:
            acc += self.read(0.5)
        return acc

    def send(self, cmd: str, wait: float = 1.5) -> str:
        os.write(self.master, (cmd + "\n").encode())
        return self.read(wait)

    def expect(self, patterns: list[str], timeout: float = 25.0) -> tuple[Optional[str], str]:
        acc = ""
        end = time.time() + timeout
        while time.time() < end:
            acc += self.read(1.0)
            for p in patterns:
                if re.search(p, acc, re.I):
                    return p, acc
        return None, acc

    def close(self) -> None:
        try:
            self.send("quit", 0.5)
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            os.close(self.master)
        except Exception:
            pass


def do_pair(mac: str, timeout: int = 40, verbose: bool = True,
            wait_inbound: float = 12.0, attempts: int = 4) -> bool:
    """Pair + trust + connect a device, answering agent prompts automatically."""
    hdr(f"Pairing {mac}")
    s = BluetoothCtl(verbose=verbose)
    try:
        s.send("power on")
        s.send("agent NoInputNoOutput")
        s.send("default-agent")
        s.send("pairable on")
        # Trust before pairing when the device is already known: a trusted device
        # gets service authorization automatically and never hits the agent timeout.
        s.send(f"trust {mac}", 1.5)
        s.send("scan on", 2.0)
        info("pairing ...")
        s.send(f"pair {mac}", 1.0)
        pat, acc = s.expect([r"Pairing successful", r"Failed to pair",
                             r"AlreadyExists"], timeout=timeout)
        if pat and "Failed" in pat:
            m = re.search(r"Failed to pair:?\s*(.*)", acc)
            fail(f"pairing failed: {m.group(1).strip() if m else 'unknown reason'}")
            s.send("scan off")
            return False
        if pat and "AlreadyExists" in pat:
            info("already paired")
        elif pat:
            ok("pairing successful")
        else:
            warn("no pairing confirmation seen; continuing to trust/connect")

        # trust right away, then let any inbound HID connection settle. The
        # controller usually connects back on its own within a few seconds.
        s.send(f"trust {mac}", 1.5)
        s.send("scan off", 1.0)
        # A bonded controller that has gone to sleep will not answer an outbound
        # page (br-connection-create-socket). Pressing its button wakes it and it
        # connects inbound on its own, so alternate waiting with connect attempts.
        print(f"\n  {c('>>> press the PS button on the controller now <<<', 'yellow')}\n")
        for attempt in range(1, attempts + 1):
            s.pump(wait_inbound)
            if is_connected(mac):          # authoritative, not stream matching
                ok("controller connected (inbound)")
                return True
            info(f"connect attempt {attempt}/{attempts} ...")
            s.send(f"connect {mac}", 1.0)
            pat, acc = s.expect([r"Connection successful", r"Connected: yes",
                                 r"Failed to connect"], timeout=timeout)
            s.pump(1.5)
            if is_connected(mac):
                ok("connected")
                return True
            m = re.search(r"Failed to connect:?\s*(.*)", acc)
            reason = m.group(1).strip() if m else "unknown reason"
            warn(f"attempt {attempt} failed ({reason})")
            if attempt < attempts:
                info("still asleep - press the PS button (light bar should light up)")
        fail("could not connect; the controller never answered")
        info("if the light bar stays off, re-pair: hold PS + Create for a fresh bond")
        return False
    finally:
        s.close()


def cmd_bt_pair(args) -> int:
    good = do_pair(args.mac, timeout=args.timeout)
    if good:
        time.sleep(2)
        pads = list_input_devices()
        if pads:
            hdr("New input devices")
            for p in pads:
                print(f"  {p['dev']:<20} {p['name']}")
            info("live test: jetson_devices.py bluetooth gamepad")
        else:
            warn("connected, but no /dev/input event device appeared yet")
            info("give it a few seconds and re-run: jetson_devices.py bluetooth status")
    return 0 if good else 1


def cmd_bt_simple(args) -> int:
    """connect / disconnect / trust / remove / info"""
    action = args.action
    rc, out, err = bt(action, args.mac, timeout=30)
    print((out or err).strip())
    return rc


def _guided_pair(name_re: re.Pattern, friendly: str, instructions: list[str],
                 args) -> int:
    hdr(f"{friendly}: guided pairing")
    for line in instructions:
        print(f"  {line}")
    try:
        input(c("\n  press Enter when the controller is in pairing mode ", "cyan"))
    except (EOFError, KeyboardInterrupt):
        print()
        return 130

    known = [d for d in bt_devices() if name_re.search(d["name"])]
    if known and not args.force_scan:
        d = known[0]
        info(f"already known: {d['mac']} ({d['name']})")
        if d.get("connected"):
            ok("already connected")
            _post_connect_report(d["mac"])
            return 0
        rc, out, _ = bt("connect", d["mac"], timeout=30)
        if "Connection successful" in out or "Connected: yes" in out:
            ok(f"connected {d['mac']}")
            _post_connect_report(d["mac"])
            return 0
        warn("reconnect failed; falling back to a fresh scan+pair")

    info(f"scanning {args.timeout}s ...")
    devs = bt_scan(args.timeout)
    matches = [d for d in devs if name_re.search(d["name"] or "")]
    if not matches:
        fail(f"no {friendly} found")
        print("\n  devices seen:")
        for d in devs:
            if d["name"]:
                print(f"    {d['mac']}  {d['name']}")
        return 1
    if len(matches) > 1:
        hdr("Multiple matches")
        for i, d in enumerate(matches):
            print(f"  {i}. {d['mac']}  {d['name']}  RSSI={d['rssi']}")
        try:
            pick = int(input(c("  select> ", "cyan")).strip() or "0")
        except (ValueError, EOFError, KeyboardInterrupt):
            pick = 0
        target = matches[max(0, min(pick, len(matches) - 1))]
    else:
        target = matches[0]
    ok(f"target: {target['mac']}  {target['name']}")
    if do_pair(target["mac"], timeout=args.timeout + 20):
        _post_connect_report(target["mac"])
        return 0
    return 1


def _post_connect_report(mac: str) -> None:
    time.sleep(2)
    rc, out, _ = bt("info", mac, timeout=15)
    hdr("Device info")
    for line in out.splitlines():
        s = line.strip()
        if any(s.startswith(k) for k in ("Name:", "Alias:", "Class:", "Icon:", "Paired:",
                                         "Trusted:", "Connected:", "Modalias:",
                                         "Battery Percentage:")):
            print(f"  {s}")
    pads = list_input_devices()
    hdr("Input devices now present")
    if not pads:
        warn("no gamepad-like /dev/input device")
    for p in pads:
        print(f"  {p['dev']:<20} {p['name']:<36} "
              f"{'readable' if p['readable'] else c('NOT READABLE', 'red')}")
    if pads:
        info("live test: jetson_devices.py bluetooth gamepad")


PS5_PAIR_STEPS = [
    f"1. turn the controller off (hold {c('PS', 'yellow')} ~10s if it is on)",
    f"2. hold {c('PS + Create', 'yellow')} (Create is left of the touchpad)",
    "3. keep holding until the light bar flashes blue in double pulses",
    "",
    c("  note: the DualSense advertises as 'Wireless Controller' or 'DualSense'",
      "grey"),
]


def find_ps5() -> Optional[dict]:
    """First known DualSense-like bond, if any."""
    for d in bt_devices():
        if PS5_NAME_RE.search(d["name"]):
            return d
    return None


def ps5_report(dev: Optional[dict]) -> None:
    hdr("PS5 controller status")
    if not dev:
        warn("no DualSense is paired with this Jetson yet")
        info("pair it from this menu, or run: jetson_devices.py bluetooth ps5 --pair")
        return
    state = []
    state.append(c("connected", "green") if dev.get("connected")
                 else c("disconnected", "yellow"))
    if dev.get("paired"):
        state.append("paired")
    if dev.get("trusted"):
        state.append("trusted")
    ok(f"{dev['name']}  {dev['mac']}")
    print(f"      state     : {' '.join(state)}")
    if dev.get("vid"):
        print(f"      usb id    : {dev['vid']}:{dev.get('pid','')}")
    _rc, out, _e = bt("info", dev["mac"], timeout=15)
    m = re.search(r"Battery Percentage:\s*(.+)", out)
    if m:
        print(f"      battery   : {m.group(1).strip()}")
    pads = [p for p in list_input_devices() if "dualsense" in p["name"].lower()
            or "wireless controller" in p["name"].lower()]
    if pads:
        for p in pads:
            print(f"      input     : {p['dev']} "
                  f"({'readable' if p['readable'] else c('NOT READABLE', 'red')})")
    elif dev.get("connected"):
        warn("connected but no input node yet - give it a second and re-check")
    else:
        print(f"      input     : {c('none (not connected)', 'grey')}")


def cmd_bt_ps5(args) -> int:
    # one-shot flags keep the command scriptable
    if args.pair:
        return _guided_pair(PS5_NAME_RE, "Sony DualSense (PS5)", PS5_PAIR_STEPS, args)
    if args.status or args.connect or args.disconnect or args.unpair or args.test:
        dev = find_ps5()
        if args.status:
            ps5_report(dev)
            return 0
        if not dev:
            fail("no DualSense is paired yet; run with --pair first")
            return 1
        if args.connect:
            return 0 if do_pair(dev["mac"], timeout=args.timeout) else 1
        if args.disconnect:
            _rc, out, err = bt("disconnect", dev["mac"], timeout=30)
            print((out or err).strip())
            return 0
        if args.unpair:
            _rc, out, err = bt("remove", dev["mac"], timeout=30)
            print((out or err).strip())
            return 0
        if args.test:
            return cmd_bt_gamepad(args)
    return ps5_menu(args)


def ps5_menu(args) -> int:
    """Interactive PS5 control panel: status, pair, connect, disconnect, test."""
    class A:
        timeout = getattr(args, "timeout", 20)
        force_scan = getattr(args, "force_scan", False)
        device = None
        duration = 0
        deadzone = getattr(args, "deadzone", 8)
        raw = False
        mac = ""

    while True:
        dev = find_ps5()
        ps5_report(dev)
        hdr("PS5 menu")
        connected = bool(dev and dev.get("connected"))
        print("  1. refresh status")
        print("  2. pair a new controller (guided)")
        print(f"  3. connect{'' if not connected else c('  (already connected)', 'grey')}")
        print("  4. disconnect")
        print("  5. unpair / forget this controller")
        print("  6. live input test (buttons, sticks, triggers)")
        print("  7. show button/axis mapping only")
        print("  q. quit")
        try:
            sel = input(c("\n  action> ", "cyan")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            if sel == "q":
                return 0
            if sel in ("1", ""):
                continue
            if sel == "2":
                _guided_pair(PS5_NAME_RE, "Sony DualSense (PS5)", PS5_PAIR_STEPS, A)
            elif sel == "3":
                if not dev:
                    warn("nothing paired yet - use option 2 first")
                elif connected:
                    info("already connected")
                else:
                    do_pair(dev["mac"], timeout=A.timeout)
            elif sel == "4":
                if not dev:
                    warn("nothing paired yet")
                else:
                    _rc, out, err = bt("disconnect", dev["mac"], timeout=30)
                    print("  " + (out or err).strip())
            elif sel == "5":
                if not dev:
                    warn("nothing paired yet")
                else:
                    confirm = input(f"  really forget {dev['mac']}? [y/N] ").strip().lower()
                    if confirm == "y":
                        _rc, out, err = bt("remove", dev["mac"], timeout=30)
                        print("  " + (out or err).strip())
                    else:
                        info("cancelled")
            elif sel == "6":
                if not connected:
                    warn("controller is not connected - use option 3 first")
                else:
                    A.duration = 0
                    cmd_bt_gamepad(A)
            elif sel == "7":
                print_ps5_mapping()
            else:
                warn("unknown action")
        except KeyboardInterrupt:
            print()
        except SystemExit as e:
            fail(str(e))
        except Exception as e:
            fail(f"{type(e).__name__}: {e}")


def print_ps5_mapping() -> None:
    hdr("PS5 controller mapping on this host")
    print("  hid_playstation is not available on this kernel, so the DualSense")
    print("  enumerates as a generic HID gamepad. Physical labels come from the")
    print("  DualSense HID report order.")
    print()
    print(f"  {'physical':<22} {'evdev name':<14} code")
    for code, (label, evname) in sorted(DUALSENSE_BTN.items()):
        print(f"  {label:<22} {evname:<14} {code}")
    print()
    for code, (label, kind) in sorted(DUALSENSE_AXES.items()):
        print(f"  {label:<22} {'ABS ' + str(code):<14} {kind}")


def cmd_bt_quest(args) -> int:
    hdr("Meta Quest 3 controllers")
    print("  " + c("Reality check:", "yellow") + " Quest 3 Touch Plus controllers bond to the")
    print("  headset over a proprietary BLE link. They do not expose a standard")
    print("  Bluetooth HID gamepad profile, so a Linux host normally cannot use them")
    print("  as a joystick even when the pairing itself succeeds.")
    print()
    print("  What this command does:")
    print("    - scans for BLE advertisements matching Quest / Oculus / Touch")
    print("    - reports vendor 0x2833 (Meta) devices")
    print("    - attempts pair + connect so you can see exactly how far it gets")
    print("    - checks whether any HID input node appeared afterwards")
    print()
    print("  Working alternatives for Quest input on the Jetson:")
    print("    - run an app on the headset that sends UDP/WebSocket poses to the Jetson")
    print("      (this repo already has that pattern in so101_unified_teleop.py)")
    print("    - use a PS5 DualSense for direct local control")
    print()
    rc = _guided_pair(
        QUEST_NAME_RE, "Meta Quest controller",
        ["1. in the headset, forget/unpair the controller you want to test",
         "2. hold the Meta/Oculus button + B (right) or Menu + Y (left) until the LED blinks",
         "3. the controller must be advertising, not bonded to the headset"],
        args)
    if rc != 0:
        info("if nothing was found, the controllers are still bonded to the headset")
    return rc


def list_input_devices() -> list[dict]:
    """Gamepad-like /dev/input/event* nodes, from /proc/bus/input/devices."""
    text = read_file("/proc/bus/input/devices")
    out: list[dict] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        handlers = ""
        evbits = ""
        uniq = ""
        m = re.search(r'N: Name="(.*)"', block)
        if m:
            name = m.group(1)
        m = re.search(r"H: Handlers=(.*)", block)
        if m:
            handlers = m.group(1)
        m = re.search(r"B: EV=([0-9a-f]+)", block)
        if m:
            evbits = m.group(1)
        m = re.search(r"U: Uniq=(.*)", block)
        if m:
            uniq = m.group(1).strip()
        ev = re.search(r"(event\d+)", handlers)
        if not ev:
            continue
        has_abs = "abs" in handlers.lower() or (int(evbits or "0", 16) & (1 << 3))
        is_js = "js" in handlers.split()
        looks_pad = is_js or (has_abs and "kbd" not in handlers) or \
            re.search(r"controller|gamepad|joystick|dualsense|xbox|wireless", name, re.I)
        if not looks_pad:
            continue
        dev = f"/dev/input/{ev.group(1)}"
        out.append({"dev": dev, "name": name, "handlers": handlers.strip(),
                    "uniq": uniq, "js": is_js,
                    "readable": os.access(dev, os.R_OK)})
    return out


class AxisPrinter:
    """
    Renders live axis values.

    Analog sticks/triggers jitter by +-1 constantly, so ignore changes below a
    deadzone. On a tty redraw one line in place; when piped (ssh, tee) \\r does
    not overwrite, so rate-limit to avoid megabytes of output.
    """

    def __init__(self, deadzone: int = 8, min_interval: float = 0.25):
        self.deadzone = deadzone
        self.min_interval = min_interval
        self.values: dict[str, int] = {}
        self.shown: dict[str, int] = {}
        self.tty = sys.stdout.isatty()
        self.last_print = 0.0

    def update(self, name: str, value: int) -> None:
        self.values[name] = value
        prev = self.shown.get(name)
        if prev is not None and abs(value - prev) < self.deadzone:
            return
        self.shown[name] = value
        now = time.time()
        if not self.tty and now - self.last_print < self.min_interval:
            return
        self.last_print = now
        line = "  ".join(f"{k}={v:>6}" for k, v in sorted(self.values.items()))
        if self.tty:
            print(f"\r  {line[:150]:<150}", end="", flush=True)
        else:
            print(f"  {line}")

    def finish(self) -> None:
        if self.tty and self.values:
            print()


# This kernel (5.15-tegra) does not ship hid_playstation, so a DualSense
# enumerates as a *generic* HID gamepad: HID buttons 1..15 land on the generic
# BTN_SOUTH..BTN_THUMBR block in report order, which is NOT the order implied by
# those evdev names. The table below is the DualSense HID report order
# (Square, Cross, Circle, Triangle, L1, R1, L2, R2, Create, Options,
#  L3, R3, PS, Touchpad, Mute).
#
# code -> (physical PS5 label, canonical evdev name)
DUALSENSE_BTN = {
    304: ("Square", "BTN_SOUTH"),
    305: ("Cross", "BTN_EAST"),
    306: ("Circle", "BTN_C"),
    307: ("Triangle", "BTN_NORTH"),
    308: ("L1", "BTN_WEST"),
    309: ("R1", "BTN_Z"),
    310: ("L2 button", "BTN_TL"),
    311: ("R2 button", "BTN_TR"),
    312: ("Create", "BTN_TL2"),
    313: ("Options", "BTN_TR2"),
    314: ("L3 stick click", "BTN_SELECT"),
    315: ("R3 stick click", "BTN_START"),
    316: ("PS", "BTN_MODE"),
    317: ("Touchpad", "BTN_THUMBL"),
    318: ("Mute", "BTN_THUMBR"),
}

# Axis layout confirmed from this host's absinfo: ABS_RX/ABS_RY rest at 0 (they
# are the analog triggers) while ABS_Z/ABS_RZ rest at ~127 (right stick).
# code -> (short name, kind); kind is "stick" (centered), "trigger" (0..max), "hat"
DUALSENSE_AXES = {
    0: ("LX", "stick"),      # ABS_X     left stick horizontal
    1: ("LY", "stick"),      # ABS_Y     left stick vertical
    2: ("RX", "stick"),      # ABS_Z     right stick horizontal
    3: ("L2", "trigger"),    # ABS_RX    left analog trigger
    4: ("R2", "trigger"),    # ABS_RY    right analog trigger
    5: ("RY", "stick"),      # ABS_RZ    right stick vertical
    16: ("D-pad X", "hat"),  # ABS_HAT0X
    17: ("D-pad Y", "hat"),  # ABS_HAT0Y
}

AXIS_NAMES = {code: name for code, (name, _k) in DUALSENSE_AXES.items()}


def button_label(code: int, ecodes: Any = None) -> str:
    """Physical PS5 label for a key code, with the kernel name alongside."""
    if code in DUALSENSE_BTN:
        label, evname = DUALSENSE_BTN[code]
        return f"{label:<16} [{evname}]"
    if ecodes is not None:
        try:
            n = ecodes.bytype[ecodes.EV_KEY][code]
            return n[0] if isinstance(n, (list, tuple)) else n
        except Exception:
            pass
    return f"btn{code}"


def button_short(code: int) -> str:
    return DUALSENSE_BTN[code][0] if code in DUALSENSE_BTN else f"btn{code}"


def joydev_test(path: str, duration: float, deadzone: int = 8) -> int:
    """
    Fallback live test over the classic joydev API (/dev/input/jsN).

    Needed because /dev/input/jsN is world-readable (0664) while event nodes are
    root:input 0660 - so this works even without the 'input' group or a logind
    uaccess ACL. Pure stdlib, no evdev needed.
    """
    import struct
    JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80
    hdr("Gamepad live test (joydev fallback)")
    try:
        f = open(path, "rb")
    except OSError as e:
        fail(f"{path}: {e}")
        return 1
    ok(f"reading {path}")
    info("press buttons / move sticks. Ctrl-C to stop.")
    print()
    printer = AxisPrinter(deadzone=deadzone)
    npress = 0
    t_end = time.time() + duration if duration else None
    try:
        while not t_end or time.time() < t_end:
            data = f.read(8)
            if not data or len(data) < 8:
                break
            _t, value, etype, number = struct.unpack("IhBB", data)
            if etype & JS_EVENT_INIT:
                continue
            if etype & JS_EVENT_BUTTON:
                printer.finish()
                if value:
                    npress += 1
                    print(f"  {c('PRESS  ', 'green')}button {number}")
                else:
                    print(f"  {c('RELEASE', 'grey')} button {number}")
            elif etype & JS_EVENT_AXIS:
                printer.update(f"ax{number}", value)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    printer.finish()
    print()
    ok(f"{npress} button presses seen, {len(printer.values)} axes moved")
    return 0


def cmd_bt_gamepad(args) -> int:
    pads = list_input_devices()
    # prefer evdev, but fall back to joydev when event nodes are unreadable
    readable_evdev = [p for p in pads if p["readable"]]
    if not readable_evdev:
        js = sorted(glob.glob("/dev/input/js*"))
        js = [j for j in js if os.access(j, os.R_OK)]
        if js:
            warn("event nodes not readable; using the joydev interface instead")
            return joydev_test(args.device if args.device in js else js[0],
                               args.duration, args.deadzone)

    evdev = require_module("evdev", "live gamepad input")
    from evdev import InputDevice, categorize, ecodes  # noqa: F401

    pads = list_input_devices()
    if args.device:
        pads = [p for p in pads if p["dev"] == args.device or args.device in p["name"]]
    if not pads:
        fail("no gamepad-like input device found")
        info("pair one first: jetson_devices.py bluetooth ps5")
        return 1
    unreadable = [p for p in pads if not p["readable"]]
    if unreadable and not any(p["readable"] for p in pads):
        fail(f"input nodes are not readable by user '{os.environ.get('USER','?')}'")
        info("fix: jetson_devices.py bluetooth fix-permissions   (then log out and back in)")
        info("or run once with: sudo -E $(which python3) jetson_devices.py bluetooth gamepad")
        return 1
    pads = [p for p in pads if p["readable"]]

    hdr("Gamepad live test")
    devices = []
    for p in pads:
        try:
            d = InputDevice(p["dev"])
            devices.append(d)
            ok(f"{p['dev']}  {d.name}  (vendor {d.info.vendor:04x} "
               f"product {d.info.product:04x})")
        except Exception as e:
            fail(f"{p['dev']}: {e}")
    if not devices:
        return 1
    return gamepad_dashboard(devices, ecodes, args)


class Dashboard:
    """
    Live PS5 controller panel.

    On a tty the panel is redrawn in place. When piped (ssh without -t, tee) the
    cursor-up trick does not work, so fall back to rate-limited event lines.
    """

    BAR = 22

    def __init__(self, name: str, raw: bool = False):
        self.name = name
        self.raw = raw
        self.tty = sys.stdout.isatty()
        self.axes: dict[int, int] = {}
        self.absinfo: dict[int, Any] = {}
        self.held: set[int] = set()
        self.last_button = ""
        self.npress = 0
        self.lines = 0
        self.last_draw = 0.0

    def norm(self, code: int, value: int) -> float:
        """Stick -> -1..1, trigger -> 0..1, hat -> -1/0/1."""
        kind = DUALSENSE_AXES.get(code, (None, "stick"))[1]
        ai = self.absinfo.get(code)
        lo, hi = (ai.min, ai.max) if ai else (0, 255)
        if hi == lo:
            return 0.0
        if kind == "hat":
            return float(value)
        if kind == "trigger":
            return (value - lo) / (hi - lo)
        mid = (hi + lo) / 2.0
        v = (value - mid) / ((hi - lo) / 2.0)
        return 0.0 if abs(v) < 0.01 else v      # snap the centre so it reads +0.00

    @staticmethod
    def _bipolar_bar(v: float, width: int = BAR) -> str:
        half = width // 2
        n = int(round(abs(v) * half))
        n = min(n, half)
        if v >= 0:
            return " " * half + "|" + "#" * n + "." * (half - n)
        return "." * (half - n) + "#" * n + "|" + " " * half

    @staticmethod
    def _unipolar_bar(v: float, width: int = BAR) -> str:
        n = min(int(round(v * width)), width)
        return "#" * n + "." * (width - n)

    @staticmethod
    def _dpad(x: float, y: float) -> str:
        active = []
        if y < 0:
            active.append("UP")
        if y > 0:
            active.append("DOWN")
        if x < 0:
            active.append("LEFT")
        if x > 0:
            active.append("RIGHT")
        glyph = f"{'<' if x < 0 else '.'} {'^' if y < 0 else '.'} " \
                f"{'v' if y > 0 else '.'} {'>' if x > 0 else '.'}"
        return f"{glyph}   {c(' + '.join(active), 'green') if active else c('centered', 'grey')}"

    def render(self) -> list[str]:
        g = lambda code: self.norm(code, self.axes.get(code, 0))  # noqa: E731
        out = [
            f"  {c(self.name, 'bold')}",
            "",
            f"  Left stick   LX {g(0):+5.2f} {self._bipolar_bar(g(0))}",
            f"               LY {g(1):+5.2f} {self._bipolar_bar(g(1))}",
            f"  Right stick  RX {g(2):+5.2f} {self._bipolar_bar(g(2))}",
            f"               RY {g(5):+5.2f} {self._bipolar_bar(g(5))}",
            f"  Triggers     L2  {g(3):4.2f} {self._unipolar_bar(g(3))}",
            f"               R2  {g(4):4.2f} {self._unipolar_bar(g(4))}",
            f"  D-pad        {self._dpad(g(16), g(17))}",
            "",
        ]
        held = "  ".join(c(button_short(b), "green") for b in sorted(self.held))
        out.append(f"  Held         {held if held else c('(none)', 'grey')}")
        out.append(f"  Last press   {self.last_button or '-'}")
        out.append(f"  Presses      {self.npress}")
        return out

    def draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_draw < 0.05:
            return
        self.last_draw = now
        if not self.tty:
            return
        if self.lines:
            sys.stdout.write(f"\033[{self.lines}A")
        lines = self.render()
        for line in lines:
            sys.stdout.write("\033[2K" + line + "\n")
        self.lines = len(lines)
        sys.stdout.flush()

    def event_line(self, text: str, rate_limit: bool = False) -> None:
        """Used in non-tty mode where the panel cannot be redrawn."""
        if self.tty:
            return
        if rate_limit:
            now = time.time()
            if now - getattr(self, "_last_axis_line", 0.0) < 0.3:
                return
            self._last_axis_line = now
        print(f"  {text}")

    def finish(self) -> None:
        if self.tty:
            self.draw(force=True)


def gamepad_dashboard(devices: list, ecodes: Any, args) -> int:
    import select as _select

    dev = devices[0]
    dash = Dashboard(dev.name, raw=getattr(args, "raw", False))
    try:
        caps = dev.capabilities(absinfo=True)
        for code, ai in caps.get(ecodes.EV_ABS, []):
            dash.absinfo[code] = ai
            dash.axes[code] = ai.value
    except Exception:
        pass

    hdr("PS5 controller mapping")
    print("  This kernel has no hid_playstation driver, so the pad enumerates as a")
    print("  generic HID gamepad. Physical labels below were resolved from the")
    print("  DualSense HID report order and this device's absinfo.")
    print()
    print(f"  {'physical':<22} {'evdev':<14} {'code'}")
    for code, (label, evname) in sorted(DUALSENSE_BTN.items()):
        print(f"  {label:<22} {evname:<14} {code}")
    for code, (label, kind) in sorted(DUALSENSE_AXES.items()):
        ai = dash.absinfo.get(code)
        rng = f"{ai.min}..{ai.max}" if ai else "?"
        print(f"  {label:<22} {'ABS ' + str(code):<14} {kind}, range {rng}")

    hdr("Live input")
    if dash.tty:
        info("move sticks / press buttons - panel updates in place. Ctrl-C to stop.")
    else:
        info("not a tty: printing events instead of a live panel "
             "(use `ssh -t` for the panel). Ctrl-C to stop.")
    print()

    fdmap = {d.fd: d for d in devices}
    t_end = time.time() + args.duration if args.duration else None
    dash.draw(force=True)
    try:
        while True:
            if t_end and time.time() > t_end:
                break
            r, _, _ = _select.select(list(fdmap), [], [], 0.2)
            dirty = False
            for fd in r:
                for e in fdmap[fd].read():
                    if e.type == ecodes.EV_KEY:
                        if e.value == 1:
                            dash.held.add(e.code)
                            dash.npress += 1
                            dash.last_button = button_label(e.code, ecodes)
                            dash.event_line(f"{c('PRESS  ', 'green')}"
                                            f"{button_label(e.code, ecodes)}")
                        elif e.value == 0:
                            dash.held.discard(e.code)
                            dash.event_line(f"{c('RELEASE', 'grey')} "
                                            f"{button_short(e.code)}")
                        dirty = True
                    elif e.type == ecodes.EV_ABS:
                        prev = dash.axes.get(e.code)
                        if prev is None or abs(e.value - prev) >= args.deadzone:
                            dash.axes[e.code] = e.value
                            dirty = True
                            summary = "  ".join(
                                f"{AXIS_NAMES.get(cd, cd)}={dash.norm(cd, v):+5.2f}"
                                for cd, v in sorted(dash.axes.items()))
                            dash.event_line(summary, rate_limit=True)
            if dirty:
                dash.draw()
    except KeyboardInterrupt:
        pass
    finally:
        for d in devices:
            try:
                d.close()
            except Exception:
                pass
    dash.finish()
    print()
    ok(f"{dash.npress} button presses seen")
    return 0


def cmd_bt_fix_permissions(args) -> int:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "cmpe"
    hdr("Input device permissions")
    rc, out, _ = run(["id", "-nG", user], timeout=10)
    groups = out.split()
    if "input" in groups:
        ok(f"user '{user}' is already in the 'input' group")
    else:
        warn(f"user '{user}' is not in the 'input' group")
        cmd = ["sudo", "usermod", "-aG", "input", user]
        print(f"  running: {' '.join(cmd)}")
        rc, out, err = run(cmd, timeout=60)
        if rc == 0:
            ok("added; log out and back in (or reboot) for it to take effect")
        else:
            fail(f"failed: {err.strip() or out.strip()}")
            return 1
    rc, _, _ = run(["lsmod"], timeout=10)
    rc2, out2, _ = run("lsmod | grep -c '^joydev'", timeout=10)
    if out2.strip() == "0":
        info("joydev module not loaded (loads automatically when a joystick appears)")
        info("preload now with: sudo modprobe joydev")
    return 0


def cmd_bt_interactive(args) -> int:
    class A:
        timeout = 15
        all = True
        force_scan = False
        device = None
        duration = 0
        deadzone = 8
        raw = False
        mac = ""

    while True:
        hdr("Bluetooth menu")
        print("  1. adapter status + known devices")
        print("  2. scan for devices")
        print("  3. pair PS5 DualSense (guided)")
        print("  4. pair Meta Quest 3 controller (guided)")
        print("  5. pair a MAC address")
        print("  6. connect a known device")
        print("  7. disconnect a known device")
        print("  8. remove (unpair) a device")
        print("  9. live gamepad input test")
        print("  p. fix /dev/input permissions")
        print("  q. quit")
        try:
            sel = input(c("\n  action> ", "cyan")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            if sel == "q":
                return 0
            elif sel == "1":
                cmd_bt_status(A)
            elif sel == "2":
                t = input("  scan seconds [15]> ").strip()
                A.timeout = int(t) if t else 15
                cmd_bt_scan(A)
            elif sel == "3":
                A.timeout = 20
                cmd_bt_ps5(A)
            elif sel == "4":
                A.timeout = 20
                cmd_bt_quest(A)
            elif sel == "5":
                A.mac = input("  MAC> ").strip().upper()
                A.timeout = 40
                if A.mac:
                    cmd_bt_pair(A)
            elif sel in ("6", "7", "8"):
                devs = bt_devices()
                if not devs:
                    warn("no known devices")
                    continue
                for i, d in enumerate(devs):
                    print(f"  {i}. {d['mac']}  {d['name']}  "
                          f"{'connected' if d.get('connected') else ''}")
                idx = input("  select> ").strip()
                if not idx.isdigit() or int(idx) >= len(devs):
                    warn("bad selection")
                    continue
                A.mac = devs[int(idx)]["mac"]
                A.action = {"6": "connect", "7": "disconnect", "8": "remove"}[sel]
                cmd_bt_simple(A)
            elif sel == "9":
                A.device = None
                A.duration = 0
                cmd_bt_gamepad(A)
            elif sel == "p":
                cmd_bt_fix_permissions(A)
            else:
                warn("unknown action")
        except KeyboardInterrupt:
            print()
        except SystemExit as e:
            fail(str(e))
        except Exception as e:
            fail(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    hdr("Host")
    print(f"  hostname : {os.uname().nodename}")
    print(f"  kernel   : {os.uname().release}")
    l4t = read_file("/etc/nv_tegra_release")
    if l4t:
        print(f"  L4T      : {l4t.splitlines()[0]}")
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  DISPLAY  : {os.environ.get('DISPLAY') or '(unset)'}")

    hdr("Tools")
    for t in ("v4l2-ctl", "gst-launch-1.0", "bluetoothctl", "hciconfig", "rfkill",
              "udevadm", "lsusb"):
        (ok if have(t) else fail)(f"{t}: {'found' if have(t) else 'MISSING'}")

    hdr("GStreamer elements")
    for e in ("nvarguscamerasrc", "nvvidconv", "v4l2src", "jpegenc", "jpegdec",
              "jpegparse", "x264enc", "ximagesink", "nveglglessink", "nvv4l2h264enc"):
        rc, _, _ = run(["gst-inspect-1.0", e], timeout=15)
        (ok if rc == 0 else warn)(f"{e}: {'yes' if rc == 0 else 'not available'}")

    hdr("Python modules (this interpreter)")
    for m in ("cv2", "numpy", "pyrealsense2", "evdev", "pygame"):
        try:
            __import__(m)
            ok(f"{m}: importable")
        except ImportError:
            warn(f"{m}: not importable")
    hdr("Other interpreters")
    for py in CANDIDATE_PYTHONS:
        if not os.path.exists(py):
            continue
        mods = []
        for m in ("cv2", "pyrealsense2", "evdev", "pygame"):
            rc, _, _ = run([py, "-c", f"import {m}"], timeout=60)
            if rc == 0:
                mods.append(m)
        print(f"  {py:<44} {', '.join(mods) or '-'}")

    cams = discover_cameras(with_formats=False)
    hdr(f"Cameras: {len(cams)}")
    for cam in cams:
        print(f"  {cam.spec:<16} {cam.label}")

    st = bt_adapter_status()
    hdr("Bluetooth")
    (ok if st["service"] == "active" else fail)(f"service: {st['service']}")
    if st.get("address"):
        ok(f"controller {st['address']} powered={st.get('powered','?')}")
    pads = list_input_devices()
    for p in pads:
        print(f"  {p['dev']:<20} {p['name']:<36} "
              f"{'readable' if p['readable'] else 'NOT READABLE'}")
    if pads and not all(p["readable"] for p in pads):
        warn("run: jetson_devices.py bluetooth fix-permissions")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jetson_devices.py",
        description="All-in-one camera and Bluetooth controller tool for Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[-1] if "Examples:" in __doc__ else None)
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = p.add_subparsers(dest="group", required=True)

    # ---- camera ----
    cam = sub.add_parser("camera", aliases=["cam"], help="camera detection and capture")
    csub = cam.add_subparsers(dest="cmd", required=True)

    def add_capture_opts(sp, defaults_out=True):
        sp.add_argument("--camera", "-c", default="0",
                        help="selector: index, /dev/videoN, csi:0, usb, rs:color, "
                             "rs:depth, rs:ir, rs:rgbd, or a name substring")
        sp.add_argument("--width", type=int, default=None)
        sp.add_argument("--height", type=int, default=None)
        sp.add_argument("--fps", type=int, default=None)
        sp.add_argument("--flip", type=int, default=None,
                        help="nvvidconv flip-method (CSI default 2 = 180 deg)")
        if defaults_out:
            sp.add_argument("--out", default=None, help="output file")
            sp.add_argument("--outdir", default=DEFAULT_OUTDIR)

    sp = csub.add_parser("list", aliases=["ls"], help="list and describe all cameras")
    sp.add_argument("--formats", "-f", action="store_true", help="show V4L2 formats")
    sp.add_argument("--all-modes", action="store_true", help="do not truncate mode lists")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_camera_list)

    sp = csub.add_parser("formats", help="show V4L2 formats for one device")
    sp.add_argument("device", help="/dev/videoN or a camera selector")
    sp.set_defaults(func=cmd_camera_formats)

    sp = csub.add_parser("snap", aliases=["snapshot"], help="save still image(s)")
    add_capture_opts(sp)
    sp.add_argument("--count", "-n", type=int, default=1)
    sp.add_argument("--warmup", type=int, default=3, help="frames to discard first")
    sp.set_defaults(func=cmd_camera_snap)

    sp = csub.add_parser("record", help="record an mp4 clip")
    add_capture_opts(sp)
    sp.add_argument("--duration", "-d", type=float, default=5.0, help="seconds")
    sp.set_defaults(func=cmd_camera_record)

    sp = csub.add_parser("show", aliases=["view", "stream"], help="live view")
    add_capture_opts(sp, defaults_out=False)
    sp.add_argument("--mode", choices=["auto", "window", "stream"], default="auto",
                    help="window = X11/EGL sink, stream = MJPEG over HTTP")
    sp.add_argument("--sink", default=None, help="explicit GStreamer video sink")
    sp.add_argument("--port", type=int, default=8090, help="MJPEG HTTP port")
    sp.set_defaults(func=cmd_camera_show)

    sp = csub.add_parser("test", help="capture smoke test on every camera")
    sp.add_argument("--camera", "-c", default=None)
    sp.add_argument("--frames", "-n", type=int, default=10)
    sp.add_argument("--timeout", type=float, default=25.0)
    sp.add_argument("--width", type=int, default=None)
    sp.add_argument("--height", type=int, default=None)
    sp.add_argument("--fps", type=int, default=None)
    sp.add_argument("--flip", type=int, default=None)
    sp.set_defaults(func=cmd_camera_test)

    sp = csub.add_parser("interactive", aliases=["menu", "i"],
                         help="interactive camera menu")
    sp.add_argument("--outdir", default=DEFAULT_OUTDIR)
    sp.add_argument("--port", type=int, default=8090)
    sp.set_defaults(func=cmd_camera_interactive)

    # ---- bluetooth ----
    blu = sub.add_parser("bluetooth", aliases=["bt"], help="Bluetooth controllers")
    bsub = blu.add_subparsers(dest="cmd", required=True)

    sp = bsub.add_parser("status", help="adapter, paired devices, input nodes")
    sp.set_defaults(func=cmd_bt_status)

    sp = bsub.add_parser("scan", help="scan for nearby devices")
    sp.add_argument("--timeout", "-t", type=int, default=15, help="scan seconds")
    sp.add_argument("--all", action="store_true", help="include unnamed devices")
    sp.set_defaults(func=cmd_bt_scan)

    sp = bsub.add_parser("devices", aliases=["list"], help="list known devices")
    sp.set_defaults(func=lambda a: cmd_bt_status(a))

    sp = bsub.add_parser("pair", help="pair + trust + connect a MAC")
    sp.add_argument("mac")
    sp.add_argument("--timeout", type=int, default=40)
    sp.set_defaults(func=cmd_bt_pair)

    for act in ("connect", "disconnect", "trust", "remove", "info"):
        sp = bsub.add_parser(act, help=f"bluetoothctl {act} <mac>")
        sp.add_argument("mac")
        sp.set_defaults(func=cmd_bt_simple, action=act)

    sp = bsub.add_parser("ps5", aliases=["dualsense"],
                         help="PS5 DualSense control panel (status/pair/connect/test)")
    sp.add_argument("--timeout", "-t", type=int, default=20)
    sp.add_argument("--force-scan", action="store_true",
                    help="scan even if the controller is already known")
    sp.add_argument("--deadzone", type=int, default=8)
    sp.add_argument("--duration", type=float, default=0)
    sp.add_argument("--device", "-d", default=None)
    sp.add_argument("--raw", action="store_true", help="show raw codes in the test")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true", help="print status and exit")
    g.add_argument("--pair", action="store_true", help="run guided pairing and exit")
    g.add_argument("--connect", action="store_true", help="connect the known pad")
    g.add_argument("--disconnect", action="store_true", help="disconnect the pad")
    g.add_argument("--unpair", action="store_true", help="forget the pad")
    g.add_argument("--test", action="store_true", help="live input test and exit")
    sp.set_defaults(func=cmd_bt_ps5)

    sp = bsub.add_parser("quest", aliases=["meta"], help="guided Meta Quest 3 controller pairing")
    sp.add_argument("--timeout", "-t", type=int, default=20)
    sp.add_argument("--force-scan", action="store_true")
    sp.set_defaults(func=cmd_bt_quest)

    sp = bsub.add_parser("gamepad", aliases=["test"], help="live controller input test")
    sp.add_argument("--device", "-d", default=None, help="/dev/input/eventN or name substring")
    sp.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl-C)")
    sp.add_argument("--deadzone", type=int, default=8,
                    help="ignore axis changes smaller than this (sticks jitter)")
    sp.add_argument("--raw", action="store_true", help="also show raw event codes")
    sp.set_defaults(func=cmd_bt_gamepad)

    sp = bsub.add_parser("fix-permissions", help="add user to the input group")
    sp.set_defaults(func=cmd_bt_fix_permissions)

    sp = bsub.add_parser("interactive", aliases=["menu", "i"], help="interactive BT menu")
    sp.set_defaults(func=cmd_bt_interactive)

    # ---- doctor ----
    sp = sub.add_parser("doctor", help="full environment + device report")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print()
        return 130
    except SystemExit as e:
        if isinstance(e.code, str):
            fail(e.code)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
