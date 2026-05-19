import RetrievalLab from "../components/RetrievalLab";

export const metadata = {
  title: "Retrieval Lab — embeddings + rerank · NVIDIA Build",
};

export default function RetrievalPage() {
  return (
    <main className="app-shell">
      <RetrievalLab />
    </main>
  );
}
