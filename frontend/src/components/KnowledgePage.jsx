import { useState, useRef, useEffect, createElement } from "react";
import { askQuestion } from "../api";
import { renderMarkdown as renderMd } from "../utils/markdown";
import "./KnowledgePage.css";

export default function KnowledgePage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function formatRagResult(data) {
    let text = data.summary || "";
    if (data.sections && data.sections.length > 0) {
      for (const section of data.sections) {
        text += `\n\n### ${section.heading}\n${section.content_markdown}`;
      }
    }
    return text || "No information available.";
  }

  async function handleSend(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q) return;

    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: q }]);

    try {
      const data = await askQuestion(q);
      const answer = formatRagResult(data);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: answer,
          sources: data.sources,
          status: data.status,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "error", text: err.message },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleChipClick(question) {
    setInput(question);
    setTimeout(() => {
      document.getElementById("knowledge-send-btn")?.click();
    }, 50);
  }

  function renderMarkdown(text) {
    return renderMd(text, { createElement, classPrefix: "k-" });
  }

  return (
    <div className="knowledge-page">
      <div className="knowledge-chat-container">
        {/* Header */}
        <div className="knowledge-chat-header">
          <div className="knowledge-chat-header-icon">🌱</div>
          <div>
            <h1 className="knowledge-chat-title">Plant Health Assistant</h1>
            <p className="knowledge-chat-subtitle">
              Ask anything about plant diseases, pests, or crop health
            </p>
          </div>
        </div>

        {/* Messages area */}
        <div className="knowledge-chat-messages">
          {/* Welcome message */}
          {messages.length === 0 && !loading && (
            <div className="knowledge-welcome">
              <div className="knowledge-welcome-icon">📖</div>
              <h2 className="knowledge-welcome-title">
                What would you like to know?
              </h2>
              <p className="knowledge-welcome-subtitle">
                Powered by the CABI knowledge base — covering diseases, pests,
                and management for a wide range of crops.
              </p>
              <div className="knowledge-welcome-chips">
                {[
                  "What are the symptoms of apple black rot?",
                  "How to prevent tomato late blight?",
                  "What causes leaf miner in tomato?",
                  "Treatment for potato early blight",
                  "How to manage grape black rot?",
                  "Corn common rust prevention methods",
                ].map((q, i) => (
                  <button key={i} className="k-chip" onClick={() => handleChipClick(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chat messages */}
          {messages.map((msg, i) => (
            <div key={i} className={`k-message k-message-${msg.role}`}>
              <div className="k-msg-avatar">
                {msg.role === "user" ? "🧑" : msg.role === "error" ? "⚠️" : "🌱"}
              </div>
              <div className="k-msg-body">
                <div className="k-msg-sender">
                  {msg.role === "user"
                    ? "You"
                    : msg.role === "error"
                    ? "Error"
                    : "Plant Health Assistant"}
                </div>
                <div className="k-msg-content">
                  {msg.role === "assistant" ? renderMarkdown(msg.text) : msg.text}
                </div>
                {/* Sources as collapsed section */}
                {msg.sources && msg.sources.length > 0 && (
                  <details className="k-sources">
                    <summary className="k-sources-title">
                      📚 {msg.sources.length} sources
                    </summary>
                    <ul className="k-sources-list">
                      {msg.sources.map((src, j) => (
                        <li key={j} className="k-source-item">
                          <span className="k-source-name">{src.title}</span>
                          <span className="k-source-meta">
                            {src.section} • {(src.score * 100).toFixed(0)}% match
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="k-message k-message-assistant">
              <div className="k-msg-avatar">🌱</div>
              <div className="k-msg-body">
                <div className="k-msg-sender">Plant Health Assistant</div>
                <div className="k-typing">
                  <span className="k-typing-dot" />
                  <span className="k-typing-dot" />
                  <span className="k-typing-dot" />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input bar */}
        <form className="knowledge-chat-input-bar" onSubmit={handleSend}>
          <input
            id="knowledge-question-input"
            type="text"
            className="k-chat-input"
            placeholder="Ask about plant diseases, pests, treatments…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <button
            id="knowledge-send-btn"
            type="submit"
            className="k-btn-send"
            disabled={!input.trim() || loading}
          >
            {loading ? (
              <span className="k-spinner" />
            ) : (
              "Ask →"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
