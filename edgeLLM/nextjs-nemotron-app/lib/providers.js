// Provider resolver — maps a model id to the right OpenAI-compatible
// endpoint, API key, and capability flags.
//
// All three providers we support speak OpenAI's /v1/chat/completions
// schema natively (NVIDIA Build is OpenAI-compatible; Anthropic exposes
// an /openai/v1/ compatibility layer; OpenAI itself). That means every
// route handler in this app can pick a model and forward unchanged.
//
// Keys are read from process.env (Next.js auto-loads .env.local). For
// the Jetson deployment we also fall back to ~/.env.local so the same
// keys that `sjsujetsontool chat` uses are picked up — see loadEnv().

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

let _homeEnvLoaded = false;

// Lazy: merge ~/.env.local into process.env on first call. Values
// already in process.env (set by Next.js from this app's .env.local)
// take precedence so per-app overrides still win.
function loadEnv() {
  if (_homeEnvLoaded) return;
  _homeEnvLoaded = true;
  const file = path.join(os.homedir(), ".env.local");
  let text;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return;
  }
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key && !(key in process.env)) {
      process.env[key] = value;
    }
  }
}

// Detect which provider a model id belongs to. Order matters — the
// first prefix that matches wins.
const PROVIDERS = [
  {
    name: "NVIDIA Build",
    keyEnv: "NVIDIA_API_KEY",
    baseUrlEnv: "NVIDIA_BASE_URL",
    baseUrlDefault: "https://integrate.api.nvidia.com/v1",
    thinking: true,
    test: (id) =>
      id.startsWith("nvidia/") ||
      id.startsWith("meta/") ||
      id.startsWith("qwen/") ||
      id.startsWith("minimaxai/") ||
      id.startsWith("mistralai/") ||
      id.startsWith("deepseek-ai/") ||
      id.startsWith("microsoft/"),
  },
  {
    name: "OpenAI",
    keyEnv: "OPENAI_API_KEY",
    baseUrlEnv: "OPENAI_BASE_URL",
    baseUrlDefault: "https://api.openai.com/v1",
    thinking: false,
    test: (id) =>
      /^gpt-/i.test(id) || id.startsWith("o1") || id.startsWith("o3") || id.startsWith("o4"),
  },
  {
    name: "Anthropic",
    keyEnv: "ANTHROPIC_API_KEY",
    baseUrlEnv: "ANTHROPIC_BASE_URL",
    // Anthropic ships an OpenAI-compatibility layer at this path.
    baseUrlDefault: "https://api.anthropic.com/v1",
    thinking: false,
    test: (id) => id.startsWith("claude-"),
  },
];

export function resolveProvider(modelId) {
  loadEnv();
  const id = String(modelId || "");
  const p = PROVIDERS.find((p) => p.test(id)) || PROVIDERS[0];
  return {
    name: p.name,
    keyEnv: p.keyEnv,
    apiKey: process.env[p.keyEnv] || "",
    baseUrl: process.env[p.baseUrlEnv] || p.baseUrlDefault,
    thinking: p.thinking,
  };
}

// Helper for routes that need to forward a key to a sidecar (e.g.
// the agent sidecar reads NVIDIA_API_KEY / SERPAPI_API_KEY from env).
export function envFromHome() {
  loadEnv();
  return process.env;
}

// ---------------------------------------------------------------------------
// Backend resolver — mirrors the `sjsujetsontool chat` backend menu.
//
// Returns { name, baseUrl, apiKey, model, keyEnv } for one of:
//
//   "llama"  — local llama.cpp server (default port 8080, started by
//              `sjsujetsontool llama bg`). No real key required.
//   "nvidia" — NVIDIA Build (cloud).
//   "openai" — OpenAI.
//   "anthropic" — Anthropic (via /openai/v1 compat layer).
//   "custom" — any OpenAI-compatible server. The caller must provide
//              `custom.baseUrl` (+ optional `custom.apiKey`, `custom.model`).
// ---------------------------------------------------------------------------

const BACKEND_DEFAULTS = {
  llama: {
    name: "Local llama.cpp",
    baseUrlEnv: "LLAMA_BASE_URL",
    baseUrlDefault: "http://localhost:8080/v1",
    keyEnv: null,
    defaultModel: "local",
  },
  node05: {
    // Our SJSU shared LLM server, exposed via the Headscale gateway. Same
    // endpoint `sjsujetsontool chat` uses for its "Our shared LLM server".
    name: "Shared SJSU llama.cpp (node05)",
    baseUrlEnv: "NODE05_BASE_URL",
    baseUrlDefault: "https://llm.forgengi.org/node05/v1",
    keyEnv: null,
    defaultModel: "Qwen3.5-9B-UD-Q6_K_XL.gguf",
  },
  nvidia: {
    name: "NVIDIA Build",
    baseUrlEnv: "NVIDIA_BASE_URL",
    baseUrlDefault: "https://integrate.api.nvidia.com/v1",
    keyEnv: "NVIDIA_API_KEY",
    defaultModel: "minimaxai/minimax-m2.7",
  },
  openai: {
    name: "OpenAI",
    baseUrlEnv: "OPENAI_BASE_URL",
    baseUrlDefault: "https://api.openai.com/v1",
    keyEnv: "OPENAI_API_KEY",
    defaultModel: "gpt-4o-mini",
  },
  anthropic: {
    name: "Anthropic",
    baseUrlEnv: "ANTHROPIC_BASE_URL",
    baseUrlDefault: "https://api.anthropic.com/v1",
    keyEnv: "ANTHROPIC_API_KEY",
    defaultModel: "claude-sonnet-4-6",
  },
  custom: {
    name: "Custom (OpenAI-compatible)",
    baseUrlEnv: "CUSTOM_BASE_URL",
    baseUrlDefault: "",
    keyEnv: "CUSTOM_API_KEY",
    defaultModel: "",
  },
};

export function resolveBackend(backendId, options = {}) {
  loadEnv();
  const id = String(backendId || "nvidia").toLowerCase();
  const def = BACKEND_DEFAULTS[id] || BACKEND_DEFAULTS.nvidia;

  // baseUrl: request override → env → built-in default
  const baseUrl =
    (options.baseUrl && String(options.baseUrl).trim()) ||
    process.env[def.baseUrlEnv] ||
    def.baseUrlDefault ||
    "";

  // apiKey: request override → env (if backend has one) → placeholder for
  // servers that ignore auth (the OpenAI Node client rejects an empty string).
  let apiKey = options.apiKey && String(options.apiKey);
  if (!apiKey && def.keyEnv) apiKey = process.env[def.keyEnv] || "";
  if (!apiKey) apiKey = "EMPTY";

  const model =
    (options.model && String(options.model).trim()) ||
    process.env.AGENT_MODEL ||
    def.defaultModel ||
    "";

  return {
    id,
    name: def.name,
    keyEnv: def.keyEnv,
    apiKey,
    baseUrl,
    model,
  };
}

// List of backends the UI shows in its dropdown. Order = display order.
export const BACKEND_MENU = [
  { id: "nvidia",    name: "NVIDIA Build (cloud)",        requiresKey: true,  defaultModel: BACKEND_DEFAULTS.nvidia.defaultModel },
  { id: "llama",     name: "Local llama.cpp (Jetson)",    requiresKey: false, defaultModel: BACKEND_DEFAULTS.llama.defaultModel },
  { id: "node05",    name: "Shared SJSU llama.cpp (node05)", requiresKey: false, defaultModel: BACKEND_DEFAULTS.node05.defaultModel },
  { id: "openai",    name: "OpenAI",                      requiresKey: true,  defaultModel: BACKEND_DEFAULTS.openai.defaultModel },
  { id: "anthropic", name: "Anthropic",                   requiresKey: true,  defaultModel: BACKEND_DEFAULTS.anthropic.defaultModel },
  { id: "custom",    name: "Custom (OpenAI-compatible)",  requiresKey: false, defaultModel: "" },
];
