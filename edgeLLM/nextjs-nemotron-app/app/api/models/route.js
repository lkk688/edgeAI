// Lists the Nemotron / NVIDIA-hosted models we want to surface in the UI.
// Kept server-side so we can swap the list without rebuilding the client.

export const dynamic = "force-dynamic";

const MODELS = [
  {
    id: "nvidia/llama-3.3-nemotron-super-49b-v1",
    label: "Nemotron Super 49B (reasoning)",
    supportsThinking: true,
  },
  {
    id: "nvidia/llama-3.1-nemotron-nano-8b-v1",
    label: "Nemotron Nano 8B (fast)",
    supportsThinking: true,
  },
  {
    id: "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    label: "Nemotron Ultra 253B (top quality)",
    supportsThinking: true,
  },
  {
    id: "meta/llama-3.3-70b-instruct",
    label: "Llama 3.3 70B Instruct (baseline)",
    supportsThinking: false,
  },
];

export async function GET() {
  return Response.json({
    default: process.env.NVIDIA_MODEL || MODELS[0].id,
    models: MODELS,
  });
}
