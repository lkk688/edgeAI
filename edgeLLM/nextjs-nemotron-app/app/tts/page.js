import TtsLab from "../components/TtsLab";

export const metadata = {
  title: "TTS Lab — zero-shot voice cloning · nvidia/magpie-tts-zeroshot",
};

export default function TtsPage() {
  return (
    <main className="app-shell">
      <TtsLab />
    </main>
  );
}
