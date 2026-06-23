# Project notes

This sample workspace is shipped with the Next.js Agent Lab. The agent
can read, grep, search, write, and edit any file under this directory —
and **only** under this directory.

## Things to try

1. **Read.** Ask the agent: *"Read calculator.py and summarize what it does."*
2. **Grep.** Ask: *"Find all TODOs in the project."*
3. **Edit.** Ask: *"Fix the typo in calculator.py."*  (There's a real one.)
4. **Web.** With `SERPAPI_API_KEY` set, ask:
   *"Search the web for the latest LangChain release and write a one-paragraph summary to webnote.md."*

## Project glossary

- **edge_agent** — the Python package providing the file tools and the
  ReAct loop.
- **AgentLab** — the React/Next.js client component that streams the trace.
- **SerpAPI** — the optional online-search backend; agent falls back to
  file tools if the key is missing.

# TODO: add more example tasks here.
