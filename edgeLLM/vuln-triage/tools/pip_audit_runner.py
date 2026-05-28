"""Tool: run `pip-audit` against a requirements file and parse its JSON.

pip-audit reads NVIDIA's PyPA advisory database and reports CVE IDs for
each affected package version. We use the `-r` mode so we never have to
actually install the vulnerable packages — pip-audit reads the file
directly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PIP_AUDIT_TIMEOUT_SEC = 120


def pip_audit_findings(requirements_path: str) -> dict[str, Any]:
    """Return a list of vulnerability findings for the packages in
    `requirements_path`.

    Result shape:

        {
          "requirements": "sample_project/requirements.txt",
          "findings": [
            {
              "package": "requests",
              "version": "2.19.1",
              "cve_ids": ["CVE-2018-18074", "GHSA-..."],
              "primary_cve": "CVE-2018-18074",
              "fix_versions": ["2.20.0"],
              "description": "..."     # short summary
            },
            ...
          ]
        }

    The agent never actually installs anything — pip-audit reads the
    requirements file in offline mode.
    """
    req_path = Path(requirements_path).resolve()
    if not req_path.is_file():
        raise FileNotFoundError(f"{requirements_path}: no such file")

    # Prefer the binary on PATH; fall back to `python -m pip_audit` so we
    # work even when pip-audit was installed via `pip install --target`
    # (which does not write a console script to PATH).
    exe_path = shutil.which("pip-audit")
    if exe_path:
        cmd = [exe_path]
    else:
        try:
            import pip_audit  # noqa: F401  pylint: disable=unused-import
        except ImportError as exc:
            raise RuntimeError(
                "pip-audit is not installed. `pip install pip-audit` first."
            ) from exc
        cmd = [sys.executable, "-m", "pip_audit"]

    proc = subprocess.run(
        [*cmd, "-r", str(req_path), "--format", "json", "--no-deps"],
        capture_output=True,
        text=True,
        timeout=PIP_AUDIT_TIMEOUT_SEC,
        check=False,
    )
    # pip-audit exits 1 when findings exist — that is the success case here.
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"pip-audit failed (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"pip-audit produced non-JSON output:\n{proc.stdout[:400]}"
        ) from exc

    findings: list[dict[str, Any]] = []
    for entry in payload.get("dependencies") or []:
        pkg = entry.get("name") or ""
        ver = entry.get("version") or ""
        for vuln in entry.get("vulns") or []:
            # `id` is usually a PYSEC- or GHSA- ID; `aliases` often lists CVE IDs.
            aliases = vuln.get("aliases") or []
            ids = [vuln.get("id"), *aliases]
            cves = sorted({i for i in ids if isinstance(i, str) and i.startswith("CVE-")})
            findings.append(
                {
                    "package": pkg,
                    "version": ver,
                    "cve_ids": cves,
                    "primary_cve": cves[0] if cves else vuln.get("id"),
                    "fix_versions": vuln.get("fix_versions") or [],
                    "description": (vuln.get("description") or "").strip().split("\n")[0],
                }
            )

    return {
        "requirements": str(req_path),
        "findings": findings,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_project/requirements.txt"
    print(json.dumps(pip_audit_findings(target), indent=2))
