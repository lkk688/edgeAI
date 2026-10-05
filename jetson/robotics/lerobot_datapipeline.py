#!/usr/bin/env python3
"""
lerobot_datapipeline.py - dataset QA, remote training submission and retrieval.

Robot-agnostic: it works on any LeRobotDataset, so the same pipeline covers the
SO-ARM101, a Gaoqing arm, a SeeedStudio re-Arm, or anything else that records in
LeRobot format. Joint names, action dimensions and camera keys are all read from
the dataset metadata rather than assumed.

Runs on the edge device (e.g. a Jetson). Training is submitted to a remote GPU
box; the transfer mechanism is pluggable so rsync+ssh can later be swapped for
something else without touching the rest of the pipeline.

    inspect     list local datasets with size and health summary
    check       per-episode quality report for one dataset
    compat      can these datasets be merged/trained together?
    merge       combine datasets via lerobot-edit-dataset
    remotes     manage remote GPU configurations
    push        send a dataset to a remote
    train       submit a training run to a remote
    status      poll a submitted run, stream its log
    fetch       pull logs and checkpoints back
    interactive guided menu over all of the above

Examples:

    ./lerobot_datapipeline.py inspect
    ./lerobot_datapipeline.py check --repo-id local/so101_pick
    ./lerobot_datapipeline.py compat --repo-id local/so101_pick --with lerobot/svla_so101_pickplace
    ./lerobot_datapipeline.py train --repo-id local/so101_pick --policy smolvla --remote cmpe28803
    ./lerobot_datapipeline.py status --run <run-id> --follow
    ./lerobot_datapipeline.py fetch --run <run-id>
    ./lerobot_datapipeline.py interactive
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

VERSION = "1.0.0"

LEROBOT_HOME = Path(os.environ.get(
    "HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
CONFIG_PATH = Path(os.environ.get(
    "LEROBOT_PIPELINE_CONFIG", Path.home() / ".lerobot_pipeline.json"))
RUNS_PATH = Path.home() / ".lerobot_runs.json"


def joint_names(meta: dict, count: int) -> list[str]:
    """Action dimension labels, taken from the dataset rather than assumed.

    Any LeRobot-format arm works: the names come from
    meta["features"]["action"]["names"], and we only fall back to positional
    labels when a dataset omits them.
    """
    names = (meta.get("features", {}).get("action", {}) or {}).get("names")
    if isinstance(names, list) and len(names) >= count:
        return [str(n) for n in names[:count]]
    return [f"dim{i}" for i in range(count)]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def c(text: str, color: str) -> str:
    if _NO_COLOR:
        return text
    codes = {"red": "31", "green": "32", "yellow": "33", "blue": "34",
             "cyan": "36", "grey": "90", "bold": "1"}
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def hdr(text: str) -> None:
    print()
    print(c(text, "bold"))
    print(c("-" * len(text), "grey"))


def ok(t: str) -> None:
    print(f"  {c('[ OK ]', 'green')} {t}")


def warn(t: str) -> None:
    print(f"  {c('[WARN]', 'yellow')} {t}")


def fail(t: str) -> None:
    print(f"  {c('[FAIL]', 'red')} {t}")


def info(t: str) -> None:
    print(f"  {c('[INFO]', 'cyan')} {t}")


def run(cmd: list[str] | str, timeout: int = 60, check: bool = False,
        capture: bool = True) -> tuple[int, str, str]:
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(cmd, shell=shell, text=True, timeout=timeout,
                           capture_output=capture)
        if check and p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "").strip())
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# remote backends - pluggable so rsync/ssh can be replaced later
# ---------------------------------------------------------------------------

@dataclass
class Remote:
    """One GPU machine. cmpe28803 is just the first entry in the config file."""
    name: str
    kind: str = "ssh"                   # selects the backend implementation
    host: str = ""                      # ssh alias or user@host
    workdir: str = "~/lerobot_runs"     # where datasets and runs live remotely
    python: str = ""                    # explicit interpreter, optional
    conda_env: str = ""                 # conda env to activate, optional
    conda_sh: str = "~/miniconda3/etc/profile.d/conda.sh"
    activate: str = ""                  # any shell snippet, e.g. "source ~/x/.venv/bin/activate"
    lerobot_home: str = "~/.cache/huggingface/lerobot"
    extra_env: dict[str, str] = field(default_factory=dict)


class RemoteBackend(ABC):
    """Transport + execution abstraction.

    Everything the pipeline needs from a remote is behind these five calls, so a
    future Slurm / Kubernetes / cloud backend only has to implement this class.
    """

    def __init__(self, remote: Remote):
        self.remote = remote

    @abstractmethod
    def check(self) -> tuple[bool, str]: ...

    @abstractmethod
    def push(self, local: Path, remote_rel: str) -> int: ...

    @abstractmethod
    def pull(self, remote_rel: str, local: Path) -> int: ...

    @abstractmethod
    def exec(self, command: str, timeout: int = 120) -> tuple[int, str, str]: ...

    @abstractmethod
    def submit(self, command: str, run_id: str) -> str: ...


class SSHRsyncBackend(RemoteBackend):
    """rsync over ssh, with nohup for detached training runs."""

    def _wrap(self, command: str) -> str:
        r = self.remote
        parts = []
        if r.conda_env:
            parts.append(f"source {r.conda_sh} && conda activate {r.conda_env}")
        if r.activate:
            parts.append(r.activate)
        for k, v in r.extra_env.items():
            parts.append(f"export {k}={shlex.quote(v)}")
        parts.append(command)
        return " && ".join(parts)

    def check(self) -> tuple[bool, str]:
        rc, out, err = self.exec("echo ok && python -c 'import lerobot,torch;"
                                 "print(lerobot.__version__, torch.__version__,"
                                 "torch.cuda.is_available())'", timeout=90)
        if rc != 0:
            return False, (err or out).strip()[:300]
        return True, out.strip().replace("ok\n", "")

    def push(self, local: Path, remote_rel: str) -> int:
        r = self.remote
        dest = f"{r.host}:{r.workdir}/{remote_rel}"
        self.exec(f"mkdir -p {r.workdir}/{Path(remote_rel).parent}", timeout=60)
        cmd = ["rsync", "-az", "--info=progress2", "--partial",
               str(local).rstrip("/") + "/", dest.rstrip("/") + "/"]
        print(f"  $ {' '.join(cmd)}")
        return subprocess.call(cmd)

    def pull(self, remote_rel: str, local: Path) -> int:
        r = self.remote
        local.mkdir(parents=True, exist_ok=True)
        src = f"{r.host}:{r.workdir}/{remote_rel}"
        cmd = ["rsync", "-az", "--info=progress2", "--partial",
               src.rstrip("/") + "/", str(local).rstrip("/") + "/"]
        print(f"  $ {' '.join(cmd)}")
        return subprocess.call(cmd)

    def exec(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        return run(["ssh", self.remote.host, self._wrap(command)], timeout=timeout)

    def submit(self, command: str, run_id: str) -> str:
        """Start detached so the run survives losing the ssh connection."""
        r = self.remote
        rundir = f"{r.workdir}/runs/{run_id}"
        script = (
            f"mkdir -p {rundir} && cd {rundir} && "
            f"cat > cmd.sh <<'SO101EOF'\n{self._wrap(command)}\nSO101EOF\n"
            f"chmod +x cmd.sh && "
            f"nohup bash cmd.sh > {rundir}/train.log 2>&1 & "
            f"echo $! > {rundir}/pid && cat {rundir}/pid"
        )
        rc, out, err = run(["ssh", r.host, script], timeout=120)
        if rc != 0:
            raise RuntimeError(f"submit failed: {(err or out).strip()[:300]}")
        return out.strip().splitlines()[-1]


BACKENDS: dict[str, type[RemoteBackend]] = {"ssh": SSHRsyncBackend}


def get_backend(remote: Remote) -> RemoteBackend:
    cls = BACKENDS.get(remote.kind)
    if cls is None:
        raise SystemExit(f"unknown remote kind '{remote.kind}'. "
                         f"known: {', '.join(BACKENDS)}")
    return cls(remote)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "remotes": {
        "cmpe28803": {
            "name": "cmpe28803",
            "kind": "ssh",
            "host": "cmpe28803",
            "workdir": "~/lerobot_runs",
            "conda_env": "lerobot",
            "conda_sh": "~/miniconda3/etc/profile.d/conda.sh",
            "lerobot_home": "~/.cache/huggingface/lerobot",
            "extra_env": {},
        },
        # Jetson Thor: 122 GB unified memory, runs the large checkpoints
        # (FLUX 3, pi05, MolmoAct2) that do not fit a 16 GB card. LeRobot main
        # in a uv venv rather than conda; F3_NATTEN_BACKEND works around
        # NATTEN's prebuilt wheels having no sm_110 kernels (needs the two-line
        # video_vae.py fix, see SO101_TRAINING.md).
        "jetsonthor": {
            "name": "jetsonthor",
            "kind": "ssh",
            "host": "jetsonthor",
            "workdir": "~/Developer/lerobot_runs",
            "activate": "source ~/Developer/lerobot/.venv/bin/activate",
            "lerobot_home": "~/.cache/huggingface/lerobot",
            "extra_env": {"F3_NATTEN_BACKEND": "flex-fna"},
        },
        # Same machine, the patched LeRobot that hqfang/pi05-so100_101 needs
        # (commit b6ec006 + the checkpoint's code/lerobot.patch).
        "jetsonthor_pi05so": {
            "name": "jetsonthor_pi05so",
            "kind": "ssh",
            "host": "jetsonthor",
            "workdir": "~/Developer/lerobot_runs",
            "activate": "source ~/Developer/lerobot-pi05so/.venv/bin/activate",
            "lerobot_home": "~/.cache/huggingface/lerobot",
            "extra_env": {},
        },
    },
    "default_remote": "cmpe28803",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_json(CONFIG_PATH, DEFAULT_CONFIG)
        info(f"wrote default config to {CONFIG_PATH}")
    cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)
    cfg.setdefault("remotes", {})
    cfg.setdefault("default_remote", next(iter(cfg["remotes"]), ""))
    return cfg


def get_remote(name: str | None) -> Remote:
    cfg = load_config()
    name = name or cfg.get("default_remote")
    entry = cfg["remotes"].get(name)
    if not entry:
        raise SystemExit(f"remote '{name}' not in {CONFIG_PATH}. "
                         f"known: {', '.join(cfg['remotes']) or '(none)'}")
    known = {f.name for f in Remote.__dataclass_fields__.values()}  # type: ignore
    return Remote(**{k: v for k, v in entry.items() if k in known})


# ---------------------------------------------------------------------------
# dataset inspection
# ---------------------------------------------------------------------------

def dataset_dir(repo_id: str, root: str | None = None) -> Path:
    return Path(root) if root else LEROBOT_HOME / repo_id


def list_datasets() -> list[str]:
    out: list[str] = []
    if not LEROBOT_HOME.exists():
        return out
    for info_file in sorted(LEROBOT_HOME.glob("*/*/meta/info.json")):
        out.append(str(info_file.parent.parent.relative_to(LEROBOT_HOME)))
    return out


def read_info(repo_id: str, root: str | None = None) -> dict | None:
    return load_json(dataset_dir(repo_id, root) / "meta" / "info.json", None)


def read_info_any(repo_id: str, root: str | None = None) -> tuple[dict | None, str]:
    """Metadata for a local dataset, falling back to the Hub.

    Only meta/info.json is fetched, so compatibility can be checked against a
    community dataset without downloading its videos.
    """
    meta = read_info(repo_id, root)
    if meta:
        return meta, "local"
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                               filename="meta/info.json")
        return load_json(Path(path), None), "hub"
    except Exception as exc:
        LOGERR = str(exc).splitlines()[0][:120]
        return None, f"unavailable ({LOGERR})"


def camera_keys(meta: dict) -> list[str]:
    return sorted(k for k in meta.get("features", {})
                  if k.startswith("observation.images."))


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1e6


def episode_table(repo_id: str, root: str | None = None) -> list[dict]:
    """Per-episode rows from meta/episodes/*.parquet, empty if unreadable."""
    base = dataset_dir(repo_id, root) / "meta" / "episodes"
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    rows: list[dict] = []
    for f in files:
        try:
            table = pq.read_table(f)
        except Exception:
            continue
        rows.extend(table.to_pylist())
    return rows


def episode_lengths(repo_id: str, root: str | None = None) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for row in episode_table(repo_id, root):
        idx = row.get("episode_index")
        length = row.get("length") or row.get("num_frames")
        if idx is not None and length is not None:
            lengths[int(idx)] = int(length)
    return lengths


def cmd_inspect(args) -> int:
    repos = [args.repo_id] if args.repo_id else list_datasets()
    if not repos:
        warn(f"no datasets under {LEROBOT_HOME}")
        info("record one first: so101_unified_teleop.py record ...")
        return 1
    hdr(f"Local datasets in {LEROBOT_HOME}")
    print(f"  {'repo_id':<34}{'eps':>5}{'frames':>9}{'fps':>5}{'MB':>8}  cameras")
    total_eps = 0
    for repo in repos:
        meta = read_info(repo)
        if not meta:
            fail(f"{repo}: no meta/info.json")
            continue
        cams = camera_keys(meta)
        total_eps += meta.get("total_episodes", 0)
        size = dir_size_mb(dataset_dir(repo))
        cam_short = ",".join(k.rsplit(".", 1)[-1] for k in cams) or c("none", "red")
        print(f"  {repo:<34}{meta.get('total_episodes',0):>5}"
              f"{meta.get('total_frames',0):>9}{meta.get('fps',0):>5}"
              f"{size:>8.0f}  {cam_short}")
    print()
    info(f"{len(repos)} dataset(s), {total_eps} episodes total")
    if total_eps < 30:
        warn(f"{total_eps} episodes is thin. LeLab suggests 30+, SmolVLA "
             f"practice is ~50 for a single task.")
    return 0


def cmd_check(args) -> int:
    repo = args.repo_id
    meta = read_info(repo, args.root)
    if not meta:
        fail(f"{repo}: no meta/info.json under {dataset_dir(repo, args.root)}")
        return 1

    hdr(f"Quality report: {repo}")
    fps = meta.get("fps", 0)
    n_eps = meta.get("total_episodes", 0)
    n_frames = meta.get("total_frames", 0)
    cams = camera_keys(meta)
    print(f"  episodes   : {n_eps}")
    print(f"  frames     : {n_frames}  ({n_frames / fps:.0f}s at {fps} fps)"
          if fps else f"  frames     : {n_frames}")
    print(f"  fps        : {fps}")
    print(f"  cameras    : {', '.join(cams) if cams else c('NONE', 'red')}")
    action = meta.get("features", {}).get("action", {})
    print(f"  action dim : {action.get('shape')}  {action.get('names')}")

    problems = 0

    if not cams:
        fail("no camera streams; vision policies (ACT, SmolVLA, pi0) cannot train on this")
        problems += 1

    lengths = episode_lengths(repo, args.root)
    if lengths:
        vals = sorted(lengths.values())
        mean = sum(vals) / len(vals)
        hdr("Episode lengths")
        print(f"  min {vals[0]}  median {vals[len(vals)//2]}  "
              f"mean {mean:.0f}  max {vals[-1]}  (frames)")
        if fps:
            print(f"  {vals[0]/fps:.1f}s .. {vals[-1]/fps:.1f}s")
        short = [i for i, v in lengths.items() if v < max(10, 0.4 * mean)]
        if short:
            warn(f"unusually short episodes (< 40% of mean): {sorted(short)}")
            info("often a mis-start or an aborted demo; consider dropping them")
            problems += 1
        spread = (vals[-1] - vals[0]) / mean if mean else 0
        if spread > 1.5:
            warn(f"episode length spread is wide ({spread:.1f}x mean)")
    else:
        info("per-episode table unreadable (needs pyarrow); skipping length check")

    # video files must exist and be non-trivial, else frames are missing
    hdr("Video files")
    vdir = dataset_dir(repo, args.root) / "videos"
    if cams and not vdir.exists():
        fail("no videos/ directory but the dataset declares cameras")
        problems += 1
    else:
        for cam in cams:
            files = sorted((vdir / cam).rglob("*.mp4")) if vdir.exists() else []
            total = sum(f.stat().st_size for f in files) / 1e6 if files else 0
            if not files:
                fail(f"{cam}: no mp4 files")
                problems += 1
            elif total < 0.05:
                fail(f"{cam}: {len(files)} file(s) but only {total:.2f} MB - likely empty")
                problems += 1
            else:
                per_frame_kb = total * 1000 / n_frames if n_frames else 0
                ok(f"{cam}: {len(files)} file(s), {total:.1f} MB "
                   f"({per_frame_kb:.1f} KB/frame)")
                if per_frame_kb < 0.5:
                    warn(f"{cam}: very low bitrate, check the frames are not blank")

    # joint statistics reveal a saturated or dead joint
    stats = load_json(dataset_dir(repo, args.root) / "meta" / "stats.json", None)
    if stats:
        hdr("Joint ranges (from meta/stats.json)")
        act = stats.get("action", {})
        mins, maxs = act.get("min"), act.get("max")
        stds = act.get("std")
        if mins and maxs:
            names = joint_names(meta, len(mins))
            ranges = [maxs[i] - mins[i] for i in range(len(mins))]
            # A joint that barely moved relative to the others means the demos
            # never exercised that degree of freedom, so the policy cannot learn
            # it no matter how many episodes you add.
            ref = sorted(ranges)[len(ranges) // 2] or 1.0
            print(f"  {'joint':<16}{'min':>9}{'max':>9}{'range':>9}{'std':>9}")
            barely = []
            for i, name in enumerate(names[:len(mins)]):
                rng = ranges[i]
                sd = stds[i] if stds else float("nan")
                flag = ""
                if rng < 1e-3:
                    flag = c("  DEAD (never moved)", "red")
                    problems += 1
                elif rng < 0.1 * ref:
                    flag = c(f"  barely used ({rng/ref:.0%} of median)", "yellow")
                    barely.append(name)
                print(f"  {name:<16}{mins[i]:>9.2f}{maxs[i]:>9.2f}"
                      f"{rng:>9.2f}{sd:>9.2f}{flag}")
            if barely:
                print()
                warn(f"barely exercised: {', '.join(barely)}")
                info("the demos never really used these joints, so a policy will")
                info("not learn to move them - vary the object/target placement")
                problems += 1
    else:
        info("no meta/stats.json; skipping joint range check")

    hdr("Verdict")
    if problems == 0:
        ok("no blocking problems found")
    else:
        warn(f"{problems} issue(s) above worth a look before training")
    if n_eps < 30:
        warn(f"only {n_eps} episodes - expect weak generalisation")
    return 0 if problems == 0 else 1


# ---------------------------------------------------------------------------
# compatibility + merging
# ---------------------------------------------------------------------------

def cmd_compat(args) -> int:
    repos = [args.repo_id] + list(args.with_repos or [])
    hdr("Merge / co-training compatibility")
    print("  LeRobot's MultiLeRobotDataset keeps only the INTERSECTION of")
    print("  features and silently disables the rest, so a camera key that is")
    print("  not shared by every dataset is dropped from training.\n")

    metas: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for repo in repos:
        meta, src = read_info_any(repo, args.root if repo == args.repo_id else None)
        if not meta:
            fail(f"{repo}: {src}")
            return 1
        metas[repo] = meta
        sources[repo] = src

    print(f"  {'repo_id':<40}{'src':>6}{'fps':>5}{'eps':>6}  cameras / action dim")
    for repo, meta in metas.items():
        cams = camera_keys(meta)
        dim = meta.get("features", {}).get("action", {}).get("shape")
        print(f"  {repo:<40}{sources[repo]:>6}{meta.get('fps',0):>5}"
              f"{meta.get('total_episodes',0):>6}  "
              f"{', '.join(k.rsplit('.',1)[-1] for k in cams) or 'none'}  | {dim}")

    feature_sets = {r: set(m.get("features", {})) for r, m in metas.items()}
    common = set.intersection(*feature_sets.values()) if feature_sets else set()
    dropped = {r: sorted(f - common) for r, f in feature_sets.items()}

    hdr("Result")
    common_cams = sorted(k for k in common if k.startswith("observation.images."))
    if not common:
        fail("no common features at all - MultiLeRobotDataset would raise")
        return 1
    if not common_cams:
        fail("no camera key is shared by all datasets")
        info("every image stream would be disabled and the policy would train blind")
        info("fix by renaming camera keys so they match, e.g. rename yours to the")
        info("community convention before merging")
    else:
        ok(f"shared camera(s): {', '.join(common_cams)}")

    for repo, keys in dropped.items():
        if keys:
            warn(f"{repo} would lose: {', '.join(keys)}")

    fps_values = {m.get("fps") for m in metas.values()}
    if len(fps_values) > 1:
        warn(f"fps differs across datasets: {sorted(fps_values)}")
        info("resample or accept that timing semantics differ between sources")
    else:
        ok(f"fps matches everywhere: {fps_values.pop()}")

    dims = {json.dumps(m.get("features", {}).get("action", {}).get("shape"))
            for m in metas.values()}
    if len(dims) > 1:
        fail(f"action dimensions differ: {dims} - not co-trainable as-is")
    else:
        ok(f"action dim matches: {dims.pop()}")
    return 0


def cmd_merge(args) -> int:
    repos = [args.repo_id] + list(args.with_repos or [])
    cmd = ["lerobot-edit-dataset", f"--repo_id={args.repo_id}",
           "--operation.type=merge",
           f"--operation.repo_ids={json.dumps(repos)}",
           f"--new_repo_id={args.new_repo_id}", "--push_to_hub=false"]
    print(f"  $ {' '.join(cmd)}\n")
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# remote training
# ---------------------------------------------------------------------------

POLICIES = {
    "act": "ACT. Trains from scratch, no pretrained weights. Strong baseline on "
           "small single-task datasets.",
    "diffusion": "Diffusion policy. From scratch.",
    "smolvla": "SmolVLA 450M. Pretrained VLA; fine-tunes on joint space directly "
               "(dynamic padding handles any action dim <= 32). ~1 GB.",
    "pi0": "pi0 3B. Larger pretrained VLA; needs more memory.",
    "pi05": "pi0.5 3B. Newer pi0 variant. ~14.5 GB.",
    "flux3": "FLUX 3 Action SO-101, 7B world-action model. ~25 GB with its "
             "encoders; LeRobot main + NATTEN only. LoRA recipe.",
    "molmoact2": "MolmoAct2 SO-100/101. Trained on community SO-100/101 data, "
                 "absolute joints, 21.8 GB fp32 / 12 GB bf16. LeRobot 0.6.1+.",
}

# Checkpoint each policy fine-tunes FROM. Using `--policy.type` for these would
# build a fresh config and leave the action head randomly initialised, which
# throws away exactly the pretraining you picked the policy for. LeRobot's own
# docs fine-tune with `--policy.path=<checkpoint>`.
PRETRAINED = {
    "smolvla": "lerobot/smolvla_base",
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
    "flux3": "black-forest-labs/flux-3-action-so101",
    "molmoact2": "lerobot/MolmoAct2-SO100_101-LeRobot",
}

# Known problems with a checkpoint as published, printed before submitting.
# See SO101_TRAINING.md "Five more candidates on Thor".
CHECKPOINT_NOTES = {
    "lerobot/MolmoAct2-SO100_101-LeRobot":
        "its config.json predates LeRobot main's field renames and fails to parse there. "
        "On the remote run `policy_bench/policy_infer.py lerobot/MolmoAct2-SO100_101-LeRobot "
        "--compat --steps 1` once, then pass --pretrained "
        "~/.cache/policy_bench/lerobot--MolmoAct2-SO100_101-LeRobot plus an explicit "
        "--rename-map (scene->cam0, wrist->cam1).",
    "hqfang/pi05-so100_101":
        "needs the patched LeRobot from its card: use --remote jetsonthor_pi05so and "
        "--extra-arg=--policy.text_tokenizer_name=<snapshot>/tokenizer/tokenizer.model "
        "(config.json points at the author's cluster).",
}

# FLUX 3 ships its own LoRA recipe; lerobot-train takes it via --config_path.
LORA_RECIPES = {"flux3": ("black-forest-labs/flux-3-action-so101", "lora.json")}


def checkpoint_camera_keys(repo_id: str) -> list[str]:
    """Image keys a pretrained checkpoint expects, read from its config.json.

    Accepts a hub repo id or a local checkpoint directory. A path that only
    exists on the remote (e.g. a --compat snapshot on Thor) cannot be read from
    here, so the camera mapping has to be given with --rename-map.
    """
    local = Path(repo_id).expanduser()
    if local.is_dir():
        cfg = load_json(local / "config.json", {})
        return [k for k in (cfg.get("input_features") or {}) if "image" in k]
    if repo_id.startswith(("/", "~", ".")):
        warn(f"{repo_id} is not a local directory (remote path?); pass --rename-map explicitly")
        return []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
        cfg = load_json(Path(path), {})
    except Exception as exc:
        warn(f"could not read {repo_id} config: {str(exc).splitlines()[0][:100]}")
        return []
    return [k for k in (cfg.get("input_features") or {}) if "image" in k]


_SCENE_HINTS = ("scene", "base", "top", "front", "desk", "overhead", "third")


def auto_rename_map(dataset_cams: list[str], ckpt_cams: list[str]) -> dict[str, str]:
    """Map dataset camera keys onto a checkpoint's keys.

    Empty when the dataset already uses the checkpoint's names. Otherwise match
    by meaning first - a wrist camera goes to the checkpoint's wrist key, a scene
    camera to its base/scene/top key, since pi0/pi05 name their inputs
    semantically - and fall back to order for the rest (SmolVLA's camera1..3
    carry no meaning at all).
    """
    if not ckpt_cams or set(dataset_cams) <= set(ckpt_cams):
        return {}
    free = list(ckpt_cams)
    mapping: dict[str, str] = {}

    def take(pred) -> str | None:
        for k in free:
            if pred(k.rsplit(".", 1)[-1].lower()):
                free.remove(k)
                return k
        return None

    for cam in dataset_cams:
        short = cam.rsplit(".", 1)[-1].lower()
        if "wrist" in short:
            hit = take(lambda n: "wrist" in n)
        elif any(h in short for h in _SCENE_HINTS):
            hit = take(lambda n: any(h in n for h in _SCENE_HINTS))
        else:
            hit = None
        if hit:
            mapping[cam] = hit
    for cam in dataset_cams:
        if cam not in mapping and free:
            mapping[cam] = free.pop(0)
    return {src: dst for src, dst in mapping.items() if src != dst}


def load_runs() -> dict:
    return load_json(RUNS_PATH, {})


def record_run(run_id: str, entry: dict) -> None:
    runs = load_runs()
    runs[run_id] = entry
    save_json(RUNS_PATH, runs)


def cmd_remotes(args) -> int:
    cfg = load_config()
    hdr(f"Remotes in {CONFIG_PATH}")
    for name, entry in cfg["remotes"].items():
        mark = " (default)" if name == cfg.get("default_remote") else ""
        print(f"  {c(name, 'bold')}{mark}")
        for k in ("kind", "host", "workdir", "conda_env"):
            if entry.get(k):
                print(f"      {k:<12} {entry[k]}")
    if args.check:
        for name in cfg["remotes"]:
            remote = get_remote(name)
            backend = get_backend(remote)
            good, detail = backend.check()
            (ok if good else fail)(f"{name}: {detail}")
    return 0


def cmd_push(args) -> int:
    remote = get_remote(args.remote)
    backend = get_backend(remote)
    local = dataset_dir(args.repo_id, args.root)
    if not local.exists():
        fail(f"{local} does not exist")
        return 1
    hdr(f"Push {args.repo_id} -> {remote.name}")
    size = dir_size_mb(local)
    info(f"{size:.0f} MB from {local}")
    rc = backend.push(local, f"datasets/{args.repo_id}")
    if rc == 0:
        ok(f"synced to {remote.host}:{remote.workdir}/datasets/{args.repo_id}")
    else:
        fail(f"rsync exited {rc}")
    return rc


def resolve_checkpoint(args) -> str | None:
    """The checkpoint to fine-tune from, or None to train from scratch."""
    if args.pretrained and args.pretrained.lower() == "none":
        return None
    return args.pretrained or PRETRAINED.get(args.policy)


def build_train_command(args, remote: Remote, run_id: str) -> str:
    ds_root = f"{remote.workdir}/datasets/{args.repo_id}"
    run_dir = f"{remote.workdir}/runs/{run_id}"
    out_dir = f"{run_dir}/outputs"
    ckpt = resolve_checkpoint(args)
    pre: list[str] = []
    parts = ["lerobot-train"]

    recipe = LORA_RECIPES.get(args.policy)
    if recipe and ckpt and not args.no_recipe:
        repo, fname = recipe
        pre.append(f"hf download {repo} {fname} --local-dir {run_dir}/recipe")
        parts.append(f"--config_path={run_dir}/recipe/{fname}")

    if ckpt:
        parts.append(f"--policy.path={ckpt}")
    else:
        parts.append(f"--policy.type={args.policy}")

    # --policy.path inherits push_to_hub from the checkpoint's config.json, and
    # when that is true lerobot-train refuses to start without --policy.repo_id.
    # Training here is local; publishing is a separate, deliberate step.
    parts.append(f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}")
    if args.push_to_hub:
        parts.append(f"--policy.repo_id={args.hub_repo_id}")

    parts += [
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={ds_root}",
        f"--output_dir={out_dir}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--job_name={run_id}",
    ]
    if args.batch_size:
        parts.append(f"--batch_size={args.batch_size}")

    rename = json.loads(args.rename_map) if args.rename_map else None
    if rename is None and ckpt:
        meta = read_info(args.repo_id, getattr(args, "root", None)) or {}
        rename = auto_rename_map(camera_keys(meta), checkpoint_camera_keys(ckpt))
    if rename:
        parts.append("--rename_map=" + shlex.quote(json.dumps(rename)))

    parts.append(f"--wandb.enable={'true' if args.wandb else 'false'}")
    parts.extend(args.extra_arg or [])
    return " && ".join(pre + [" ".join(parts)])


def describe_training(args) -> None:
    """Say plainly whether this is a fine-tune or from-scratch, and why."""
    ckpt = resolve_checkpoint(args)
    if ckpt:
        info(f"fine-tuning from {c(ckpt, 'bold')} (--policy.path)")
        if ckpt in CHECKPOINT_NOTES:
            warn(f"{ckpt}: {CHECKPOINT_NOTES[ckpt]}")
        meta = read_info(args.repo_id, getattr(args, "root", None)) or {}
        ours = camera_keys(meta)
        theirs = checkpoint_camera_keys(ckpt)
        if theirs:
            print(f"      dataset cameras   : {', '.join(k.rsplit('.',1)[-1] for k in ours) or 'none'}")
            print(f"      checkpoint expects: {', '.join(k.rsplit('.',1)[-1] for k in theirs)}")
            rename = (json.loads(args.rename_map) if args.rename_map
                      else auto_rename_map(ours, theirs))
            if not rename:
                ok("camera keys already match, no rename needed")
            else:
                for src, dst in rename.items():
                    print(f"      rename {src.rsplit('.',1)[-1]:>8} -> {dst.rsplit('.',1)[-1]}")
    else:
        info(f"training {args.policy} from scratch (--policy.type)")


def cmd_train(args) -> int:
    if getattr(args, "push_to_hub", False) and not getattr(args, "hub_repo_id", ""):
        fail("--push-to-hub needs --hub-repo-id, e.g. myuser/so101_smolvla")
        return 1
    remote = get_remote(args.remote)
    backend = get_backend(remote)

    hdr(f"Submit training to {remote.name}")
    good, detail = backend.check()
    if not good:
        fail(f"remote not usable: {detail}")
        return 1
    ok(f"remote ready: {detail}")

    if not args.skip_push:
        local = dataset_dir(args.repo_id, args.root)
        if not local.exists():
            fail(f"{local} does not exist locally")
            return 1
        info("syncing dataset ...")
        if backend.push(local, f"datasets/{args.repo_id}") != 0:
            fail("dataset sync failed")
            return 1

    run_id = args.run_id or (
        f"{args.repo_id.replace('/', '_')}-{args.policy}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}")
    describe_training(args)
    command = build_train_command(args, remote, run_id)

    print()
    info(f"run id: {c(run_id, 'bold')}")
    print(f"  $ {command}\n")
    if args.dry_run:
        info("dry run, nothing submitted")
        return 0

    pid = backend.submit(command, run_id)
    record_run(run_id, {
        "run_id": run_id, "remote": remote.name, "pid": pid,
        "repo_id": args.repo_id, "policy": args.policy,
        "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": command,
    })
    ok(f"submitted, remote pid {pid}")
    info(f"follow it:  {Path(sys.argv[0]).name} status --run {run_id} --follow")
    info(f"fetch it:   {Path(sys.argv[0]).name} fetch --run {run_id}")
    return 0


def cmd_status(args) -> int:
    runs = load_runs()
    if not args.run:
        hdr("Submitted runs")
        if not runs:
            info("none recorded yet")
        for rid, entry in sorted(runs.items(), key=lambda kv: kv[1].get("submitted", "")):
            print(f"  {rid}")
            print(f"      remote {entry.get('remote')}  policy {entry.get('policy')}"
                  f"  submitted {entry.get('submitted')}")
        return 0

    entry = runs.get(args.run)
    if not entry:
        fail(f"unknown run '{args.run}'")
        return 1
    remote = get_remote(entry["remote"])
    backend = get_backend(remote)
    rundir = f"{remote.workdir}/runs/{args.run}"

    rc, out, _ = backend.exec(
        f"if kill -0 $(cat {rundir}/pid) 2>/dev/null; then echo RUNNING; "
        f"else echo FINISHED; fi; "
        f"ls {rundir}/outputs/checkpoints 2>/dev/null | tail -3", timeout=60)
    state = "UNKNOWN"
    checkpoints: list[str] = []
    for line in out.splitlines():
        if line.strip() in ("RUNNING", "FINISHED"):
            state = line.strip()
        elif line.strip():
            checkpoints.append(line.strip())

    hdr(f"Run {args.run}")
    print(f"  remote     : {remote.name} ({remote.host})")
    print(f"  policy     : {entry.get('policy')}   dataset: {entry.get('repo_id')}")
    print(f"  state      : {c(state, 'green' if state == 'RUNNING' else 'yellow')}")
    print(f"  checkpoints: {', '.join(checkpoints) if checkpoints else '(none yet)'}")

    if args.follow:
        hdr("Log (Ctrl-C to stop following)")
        try:
            subprocess.call(["ssh", remote.host, f"tail -f {rundir}/train.log"])
        except KeyboardInterrupt:
            print()
    else:
        rc, out, _ = backend.exec(f"tail -{args.lines} {rundir}/train.log", timeout=60)
        hdr(f"Last {args.lines} log lines")
        print(out or "  (empty)")
    return 0


def cmd_fetch(args) -> int:
    runs = load_runs()
    entry = runs.get(args.run)
    if not entry:
        fail(f"unknown run '{args.run}'")
        return 1
    remote = get_remote(entry["remote"])
    backend = get_backend(remote)
    dest = Path(args.dest or (Path.home() / "lerobot_results" / args.run))
    hdr(f"Fetch {args.run} -> {dest}")
    what = "outputs" if not args.logs_only else "outputs/train.log"
    rc = backend.pull(f"runs/{args.run}/{'' if not args.logs_only else ''}", dest)
    if rc == 0:
        ok(f"pulled into {dest}")
        for p in sorted(dest.rglob("*.safetensors"))[:10]:
            print(f"      {p.relative_to(dest)}  {p.stat().st_size/1e6:.0f} MB")
    else:
        fail(f"rsync exited {rc}")
    return rc


# ---------------------------------------------------------------------------
# interactive menu
# ---------------------------------------------------------------------------

def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        return input(f"  {text}{suffix}: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def pick(items: list[str], what: str) -> str | None:
    if not items:
        warn(f"no {what} available")
        return None
    for i, item in enumerate(items):
        print(f"    {i}. {item}")
    choice = prompt(f"select {what}", "0")
    try:
        return items[int(choice)]
    except (ValueError, IndexError):
        warn("bad selection")
        return None


def cmd_interactive(args) -> int:
    cfg = load_config()

    class A:
        repo_id = None
        root = None
        remote = cfg.get("default_remote")
        with_repos: list[str] = []
        new_repo_id = ""
        dry_run = False
        check = False
        run = None
        follow = False
        lines = 40
        dest = None
        logs_only = False
        policy = "smolvla"
        steps = 20000
        batch_size = None
        save_freq = 5000
        wandb = False
        pretrained = ""
        rename_map = ""
        no_recipe = False
        push_to_hub = False
        hub_repo_id = ""
        skip_push = False
        run_id = ""
        extra_arg: list[str] = []

    while True:
        datasets = list_datasets()
        runs = load_runs()
        print("\n" + "=" * 66)
        print("  LeRobot data + training pipeline")
        print("=" * 66)
        print(f"  datasets : {len(datasets)}      remote: {A.remote}")
        print(f"  runs     : {len(runs)}")
        print()
        print("  1. inspect all datasets")
        print("  2. quality check one dataset")
        print("  3. check merge compatibility (local + community)")
        print("  4. merge datasets")
        print("  5. remotes: list / test")
        print("  6. push a dataset to a remote")
        print("  7. submit a training run")
        print("  8. run status / logs")
        print("  9. fetch results")
        print("  q. quit")

        choice = prompt("action").lower()
        try:
            if choice in ("q", "quit", "exit"):
                return 0
            if choice == "1":
                A.repo_id = None
                cmd_inspect(A)
            elif choice == "2":
                A.repo_id = pick(datasets, "dataset")
                if A.repo_id:
                    cmd_check(A)
            elif choice == "3":
                A.repo_id = pick(datasets, "base dataset")
                if A.repo_id:
                    others = prompt("other repo_ids (comma separated)")
                    A.with_repos = [x.strip() for x in others.split(",") if x.strip()]
                    cmd_compat(A)
            elif choice == "4":
                A.repo_id = pick(datasets, "base dataset")
                if A.repo_id:
                    others = prompt("datasets to merge in (comma separated)")
                    A.with_repos = [x.strip() for x in others.split(",") if x.strip()]
                    A.new_repo_id = prompt("new repo_id", "local/merged")
                    A.dry_run = prompt("dry run? y/n", "y").lower().startswith("y")
                    cmd_merge(A)
            elif choice == "5":
                A.check = prompt("test connectivity? y/n", "y").lower().startswith("y")
                cmd_remotes(A)
                A.check = False
            elif choice == "6":
                A.repo_id = pick(datasets, "dataset")
                if A.repo_id:
                    A.remote = pick(list(cfg["remotes"]), "remote") or A.remote
                    cmd_push(A)
            elif choice == "7":
                A.repo_id = pick(datasets, "dataset")
                if not A.repo_id:
                    continue
                A.remote = pick(list(cfg["remotes"]), "remote") or A.remote
                print("\n  policies:")
                for name, desc in POLICIES.items():
                    print(f"    {name:<10} {desc}")
                A.policy = prompt("policy", "smolvla")
                A.steps = int(prompt("steps", "20000"))
                bs = prompt("batch size (blank = checkpoint default)", "")
                A.batch_size = int(bs) if bs else None
                A.dry_run = prompt("dry run? y/n", "y").lower().startswith("y")
                cmd_train(A)
                A.dry_run = False
            elif choice == "8":
                A.run = pick(sorted(runs), "run") if runs else None
                A.follow = bool(A.run) and prompt("follow log? y/n", "n").lower().startswith("y")
                cmd_status(A)
            elif choice == "9":
                A.run = pick(sorted(runs), "run") if runs else None
                if A.run:
                    cmd_fetch(A)
            else:
                warn("unknown action")
        except KeyboardInterrupt:
            print()
        except SystemExit as exc:
            fail(str(exc))
        except Exception as exc:
            fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lerobot_datapipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect", help="list local datasets")
    sp.add_argument("--repo-id", default=None)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("check", help="per-episode quality report")
    sp.add_argument("--repo-id", required=True)
    sp.add_argument("--root", default=None)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("compat", help="can these be merged / co-trained?")
    sp.add_argument("--repo-id", required=True)
    sp.add_argument("--with", dest="with_repos", nargs="+", required=True)
    sp.add_argument("--root", default=None)
    sp.set_defaults(func=cmd_compat)

    sp = sub.add_parser("merge", help="merge datasets with lerobot-edit-dataset")
    sp.add_argument("--repo-id", required=True)
    sp.add_argument("--with", dest="with_repos", nargs="+", required=True)
    sp.add_argument("--new-repo-id", required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_merge)

    sp = sub.add_parser("remotes", help="list remote GPU configs")
    sp.add_argument("--check", action="store_true", help="test connectivity")
    sp.set_defaults(func=cmd_remotes)

    sp = sub.add_parser("push", help="send a dataset to a remote")
    sp.add_argument("--repo-id", required=True)
    sp.add_argument("--remote", default=None)
    sp.add_argument("--root", default=None)
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("train", help="submit a training run to a remote")
    sp.add_argument("--repo-id", required=True)
    sp.add_argument("--remote", default=None)
    sp.add_argument("--policy", default="smolvla", choices=sorted(POLICIES))
    sp.add_argument("--steps", type=int, default=20000)
    sp.add_argument("--batch-size", type=int, default=None,
                    help="omit to keep the checkpoint's / recipe's own value")
    sp.add_argument("--save-freq", type=int, default=5000)
    sp.add_argument("--pretrained", default="",
                    help="checkpoint to fine-tune from; default per policy, "
                         "'none' trains from scratch")
    sp.add_argument("--rename-map", default="",
                    help="JSON dataset->checkpoint camera key map; auto if omitted")
    sp.add_argument("--no-recipe", action="store_true",
                    help="skip a policy's bundled LoRA recipe (flux3)")
    sp.add_argument("--push-to-hub", action="store_true",
                    help="publish the trained policy (needs --hub-repo-id)")
    sp.add_argument("--hub-repo-id", default="", help="e.g. myuser/so101_smolvla")
    sp.add_argument("--wandb", action="store_true")
    sp.add_argument("--skip-push", action="store_true", help="dataset already synced")
    sp.add_argument("--run-id", default="")
    sp.add_argument("--root", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--extra-arg", action="append")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("status", help="poll a run / list runs")
    sp.add_argument("--run", default=None)
    sp.add_argument("--follow", action="store_true")
    sp.add_argument("--lines", type=int, default=40)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("fetch", help="pull logs and checkpoints back")
    sp.add_argument("--run", required=True)
    sp.add_argument("--dest", default=None)
    sp.add_argument("--logs-only", action="store_true")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("interactive", aliases=["menu", "i"], help="guided menu")
    sp.set_defaults(func=cmd_interactive)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print()
        return 130
    except SystemExit as exc:
        if isinstance(exc.code, str):
            fail(exc.code)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
