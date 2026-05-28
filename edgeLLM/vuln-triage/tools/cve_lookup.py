"""Tool: fetch CVE metadata from NIST's public NVD JSON API.

The free-tier endpoint allows ~5 requests per 30 seconds without an API
key. For this tutorial we cache successful lookups in
`./.cve_cache/<cve_id>.json` so re-running the agents costs zero quota.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CACHE_DIR = Path(os.environ.get("VULN_TRIAGE_CACHE", ".cve_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(cve_id: str) -> Path:
    return CACHE_DIR / f"{cve_id.upper()}.json"


def lookup_cve(cve_id: str) -> dict[str, Any]:
    """Return a trimmed dict describing `cve_id`.

    Tries the local cache first, then NVD. Result shape:

        {
          "cve_id":      "CVE-2018-18074",
          "description": "...",
          "cvss":        {"version": "3.0", "score": 8.1, "severity": "HIGH"},
          "references":  ["https://..."],   # first 4
          "cwe_ids":     ["CWE-200"],
          "vulnerable_specifiers": [
              {"package": "requests", "version_end_excluding": "2.20.0"}
          ]
        }
    """
    cve_id = cve_id.strip().upper()
    cached = _cache_path(cve_id)
    if cached.exists():
        return json.loads(cached.read_text())

    headers: dict[str, str] = {}
    if api_key := os.environ.get("NVD_API_KEY"):
        headers["apiKey"] = api_key

    # NVD will occasionally 503; one retry with backoff is plenty.
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=15.0, trust_env=False) as client:
                resp = client.get(NVD_URL, params={"cveId": cve_id}, headers=headers)
                if resp.status_code == 404:
                    raise LookupError(f"NVD has no record of {cve_id}")
                resp.raise_for_status()
                payload = resp.json()
            break
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            if attempt == 0:
                time.sleep(2.0)
    else:
        raise RuntimeError(f"NVD lookup for {cve_id} failed: {last_exc}")

    vulns = payload.get("vulnerabilities") or []
    if not vulns:
        raise LookupError(f"NVD returned no `vulnerabilities` array for {cve_id}")
    cve = vulns[0].get("cve", {})

    description = ""
    for desc in cve.get("descriptions") or []:
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    cvss: dict[str, Any] = {}
    metrics = cve.get("metrics") or {}
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            m = metrics[key][0].get("cvssData", {})
            cvss = {
                "version": m.get("version"),
                "score": m.get("baseScore"),
                "severity": (
                    metrics[key][0].get("baseSeverity")
                    or m.get("baseSeverity")
                    or ""
                ),
                "vector": m.get("vectorString"),
            }
            break

    references = [r.get("url") for r in (cve.get("references") or [])][:4]
    cwe_ids: list[str] = []
    for w in cve.get("weaknesses") or []:
        for d in w.get("description") or []:
            if d.get("lang") == "en" and d.get("value", "").startswith("CWE-"):
                cwe_ids.append(d["value"])

    # Best-effort: enumerate affected (package, version range) tuples
    # from CPE strings. The full CPE 2.3 grammar is overkill here — we
    # parse only the bits a triage prompt needs.
    specifiers: list[dict[str, Any]] = []
    for cfg in cve.get("configurations") or []:
        for node in cfg.get("nodes") or []:
            for match in node.get("cpeMatch") or []:
                cpe = (match.get("criteria") or "").split(":")
                # cpe:2.3:a:vendor:product:version:...
                if len(cpe) > 5 and match.get("vulnerable"):
                    spec = {
                        "package": cpe[4],
                        "version": cpe[5] if cpe[5] != "*" else None,
                    }
                    for kk in (
                        "versionStartIncluding",
                        "versionStartExcluding",
                        "versionEndIncluding",
                        "versionEndExcluding",
                    ):
                        if kk in match:
                            spec[kk] = match[kk]
                    specifiers.append(spec)

    out = {
        "cve_id": cve_id,
        "description": description,
        "cvss": cvss,
        "references": references,
        "cwe_ids": cwe_ids,
        "vulnerable_specifiers": specifiers[:6],
    }
    cached.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "CVE-2018-18074"
    print(json.dumps(lookup_cve(target), indent=2))
