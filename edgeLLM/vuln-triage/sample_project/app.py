"""Tiny demo app used by the vulnerability-triage tutorial.

The triage agent reads this file via the `read_file` tool and uses the
patterns below to decide whether each CVE in requirements.txt is reachable.

Design notes (kept verbose so students reading this file can see the
reasoning the LLM is supposed to reproduce):

  - `requests` is used directly with a caller-supplied URL → the
    vulnerable code path (HTTP-based attacks) is reachable.
  - `jinja2` is imported, but the template string is a constant defined
    in this file. The sandbox-escape class of CVEs is not reachable
    because no user-controlled template text ever reaches Jinja.
  - `pyyaml` is listed in requirements.txt but is *not imported anywhere
    in this code base* — pure dead weight.
  - `urllib3` is only ever called transitively by requests; no direct
    usage in this file.
"""

from __future__ import annotations

import jinja2
import requests

# Hard-coded template defined at import time. There is no public API on
# this module that lets a user override it — the agent should notice.
_STATUS_TEMPLATE = jinja2.Template("Status for {{ url }}: {{ status }}")


def fetch_status(url: str) -> dict:
    """Return the HTTP status and body length for `url`.

    The `url` argument is forwarded straight into `requests.get`, with no
    scheme validation or proxy handling. Any caller can supply an
    arbitrary string here.
    """
    response = requests.get(url, timeout=5)
    return {"status": response.status_code, "length": len(response.content)}


def render_status(url: str, status: int) -> str:
    """Render the fixed template above. No user input ever reaches Jinja
    other than as a plain `{{ var }}` substitution into a trusted string."""
    return _STATUS_TEMPLATE.render(url=url, status=status)


if __name__ == "__main__":
    info = fetch_status("https://example.com")
    print(render_status("https://example.com", info["status"]))
