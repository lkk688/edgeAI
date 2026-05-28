# sample_project — intentionally vulnerable

Tiny Python "application" used by the vulnerability-triage tutorials
(`docs/curriculum/12*`).

| File              | What it shows the agent |
|-------------------|-------------------------|
| `requirements.txt`| 4 packages pinned to versions with known CVEs |
| `app.py`          | Three deliberate usage patterns: directly-exploited (`requests`), present but safely-used (`jinja2`), declared-but-unused (`pyyaml`) |

Do **not** run `pip install -r requirements.txt` from this folder in your
real environment — install it inside an isolated `venv` if you want to
try `pip-audit` locally.
