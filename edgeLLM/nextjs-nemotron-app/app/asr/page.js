import AsrLab from "../components/AsrLab";

export const metadata = {
  title: "ASR Lab — speech-to-text · nvidia/nemotron-asr-streaming",
};

export default function AsrPage() {
  return (
    <main className="app-shell">
      <AsrLab />
    </main>
  );
}
