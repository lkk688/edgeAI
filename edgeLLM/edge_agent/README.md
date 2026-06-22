# edge-agent

A tiny, dependency-free **ReAct agent loop** plus a safe **file-tool kit** — the
core of an AI agent, small enough to read in one sitting. Built for the
[SJSU Edge AI curriculum](https://lkk688.github.io/edgeAI/) and used by
`sjsujetsontool chat --agent`.

📖 Tutorial: <https://lkk688.github.io/edgeAI/curriculum/13_react_agent/>

## What's inside

| Module | What it is |
|---|---|
| `edge_agent.tools` | `Tools` — `read_file` / `grep` / `search_files` / `write_file` / `edit_file`, confined to a project root. |
| `edge_agent.react_loop` | `ReActAgent` — the Reason+Act text-protocol loop; works on **any** chat model. |
| `edge_agent.tool_calling` | `run_tool_calling()` — same tools via the provider's **native** `tools=` field (needs `openai`). |

## Install

```bash
pip install -e edgeLLM/edge_agent              # editable, from the repo
# or with native tool-calling support:
pip install -e "edgeLLM/edge_agent[toolcalling]"
```

## Use

```python
from edge_agent import ReActAgent, Tools

def complete(messages):
    # send messages to any OpenAI-compatible model, return the assistant text
    ...

agent = ReActAgent(complete, Tools(root="./my_project"))
print(agent.run("Read app.py and tell me what it does."))
```

The ReAct loop is decoupled from any HTTP client — you pass a
`complete(messages) -> str` callable, so the same agent runs on llama.cpp,
NVIDIA Build, OpenAI, or Anthropic.

Native tool-calling demo (CLI installed as `edge-agent`):

```bash
edge-agent "Summarize app.py and list risky calls" --dir ./sample_project
```
