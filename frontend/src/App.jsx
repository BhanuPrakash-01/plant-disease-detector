import { useState } from "react";
import DiagnosisPage from "./components/DiagnosisPage";
import KnowledgePage from "./components/KnowledgePage";
import "./App.css";

const TABS = [
  { id: "diagnosis", label: "🔬 Image Diagnosis", icon: "🔬" },
  { id: "knowledge", label: "🌱 Plant Knowledge", icon: "🌱" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("diagnosis");

  return (
    <div className="app">
      {/* ── Top Nav ────────────────────────────────────────────── */}
      <nav className="top-nav">
        <div className="nav-inner">
          <div className="nav-brand">
            <span className="brand-icon">🌿</span>
            <span className="brand-text">Plant Disease AI</span>
          </div>
          <div className="nav-tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* ── Page Content ──────────────────────────────────────── */}
      <main className="main-content">
        {activeTab === "diagnosis" && <DiagnosisPage />}
        {activeTab === "knowledge" && <KnowledgePage />}
      </main>

      {/* ── Footer ────────────────────────────────────────────── */}
      <footer className="app-footer">
        <p>
          Powered by EfficientNet · LIME · CABI Knowledge Base · Gemini
        </p>
      </footer>
    </div>
  );
}
