"""Tools the vuln-triage agents can call.

Each tool is a plain Python function with a JSON-serialisable return
value. The agents wrap them in OpenAI-compatible tool schemas before
sending them to the model.
"""

from .cve_lookup import lookup_cve  # noqa: F401
from .code_search import read_file, search_usage  # noqa: F401
from .pip_audit_runner import pip_audit_findings  # noqa: F401
