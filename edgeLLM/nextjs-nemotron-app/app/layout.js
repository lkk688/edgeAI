import "./globals.css";
import NavBar from "./components/NavBar";

export const metadata = {
  title: "Next.js + NVIDIA Nemotron — Edge AI Tutorial",
  description:
    "A multi-lab Next.js app for Jetson students: streaming chat, retrieval, and omni multimodal — all powered by NVIDIA Build.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        {children}
      </body>
    </html>
  );
}
