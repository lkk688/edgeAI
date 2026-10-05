"""Import every LeRobot policy and build its default config.

INSTALL_OK only proves pip resolved; this proves the code actually loads on this
machine (aarch64 wheels, optional native kernels like natten / flash-attn).
"""
import importlib, sys, traceback, warnings
warnings.filterwarnings("ignore")

POLICIES = ["act", "diffusion", "smolvla", "pi0", "pi05", "pi0_fast", "groot",
            "xvla", "wall_x", "molmoact2", "eo1", "evo1", "vla_jepa", "fastwam",
            "lawam", "lingbot_va", "multi_task_dit", "gaussian_actor", "flux3"]

from lerobot.policies.factory import get_policy_class, make_policy_config

rows = []
for name in POLICIES:
    status, detail = "OK", ""
    try:
        cls = get_policy_class(name)
        cfg = make_policy_config(name)
        detail = f"{cls.__name__} / {type(cfg).__name__}"
    except Exception as e:
        status = "FAIL"
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else ""
        detail = f"{type(e).__name__}: {msg}"[:150]
    rows.append((name, status, detail))
    print(f"  {name:<16} {status:<5} {detail}", flush=True)

ok = sum(r[1] == "OK" for r in rows)
print(f"\n{ok}/{len(rows)} policies import and build a default config")
