// Models surfaced in the UI dropdown. The chat route infers the provider
// (NVIDIA / OpenAI / Anthropic) from the model id, so you can mix them here —
// each just needs its key in ~/.env.local. Kept server-side so the list can
// change without rebuilding the client.

export const dynamic = "force-dynamic";

const MODELS = [
  // --- NVIDIA Build  (NVIDIA_API_KEY) ---
  { id: "nvidia/llama-3.3-nemotron-super-49b-v1",  label: "NVIDIA Nemotron Super 49B (reasoning)", supportsThinking: true },
  { id: "nvidia/llama-3.1-nemotron-nano-8b-v1",    label: "NVIDIA Nemotron Nano 8B (fast)",        supportsThinking: true },
  { id: "nvidia/llama-3.1-nemotron-ultra-253b-v1", label: "NVIDIA Nemotron Ultra 253B (top)",      supportsThinking: true },
  // --- OpenAI  (OPENAI_API_KEY) ---
  { id: "gpt-4o-mini", label: "OpenAI GPT-4o mini", supportsThinking: false },
  { id: "gpt-4o",      label: "OpenAI GPT-4o",      supportsThinking: false },
  // --- Anthropic Claude, OpenAI-compatible endpoint  (ANTHROPIC_API_KEY) ---
  { id: "claude-haiku-4-5",  label: "Anthropic Claude Haiku 4.5",  supportsThinking: false },
  { id: "claude-sonnet-4-6", label: "Anthropic Claude Sonnet 4.6", supportsThinking: false },
];

export async function GET() {
  return Response.json({
    default: process.env.NVIDIA_MODEL || MODELS[0].id,
    models: MODELS,
  });
}
