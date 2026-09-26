import { useState, useRef, useEffect, createElement } from "react";
import { analyzeImage, askFollowUp } from "../api";
import { renderMarkdown as renderMd } from "../utils/markdown";
import "./DiagnosisPage.css";

export default function DiagnosisPage() {
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Chat follow-up state
  const [chatMessages, setChatMessages] = useState([]);
  const [followUpInput, setFollowUpInput] = useState("");
  const [followUpLoading, setFollowUpLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Active tab for results panel
  const [activeResultTab, setActiveResultTab] = useState("analysis");

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, followUpLoading]);

  function handleImageChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setChatMessages([]);
    setActiveResultTab("analysis");
  }

  function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setChatMessages([]);
    setActiveResultTab("analysis");
  }

  function clearImage() {
    setImageFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setChatMessages([]);
  }

  async function handleAnalyze() {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setChatMessages([]);
    try {
      const data = await analyzeImage(imageFile, null);
      setResult(data);
      setActiveResultTab("analysis");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFollowUp(e) {
    e.preventDefault();
    if (!followUpInput.trim() || !result?.prediction) return;

    const userQuestion = followUpInput.trim();
    setFollowUpInput("");
    setFollowUpLoading(true);

    setChatMessages((prev) => [
      ...prev,
      { role: "user", text: userQuestion },
    ]);

    try {
      const data = await askFollowUp(
        userQuestion,
        result.prediction.predicted_class,
        result.prediction.confidence
      );
      const answer = formatRagAnswer(data.rag);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: answer, rag: data.rag },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { role: "error", text: err.message },
      ]);
    } finally {
      setFollowUpLoading(false);
    }
  }

  function formatClassName(cls) {
    if (!cls) return "";
    return cls.replace(/___/g, " — ").replace(/_/g, " ");
  }

  function formatRagAnswer(rag) {
    if (!rag) return "No information available.";
    let text = rag.summary || "";
    if (rag.sections && rag.sections.length > 0) {
      for (const section of rag.sections) {
        text += `\n\n### ${section.heading}\n${section.content_markdown}`;
      }
    }
    return text || "No information available.";
  }

  function renderMarkdown(text) {
    return renderMd(text, { createElement });
  }

  const hasPrediction = result?.prediction;
  const hasAnalysis = result?.rag;

  return (
    <div className="diagnosis-page">
      <div className="diagnosis-header">
        <h1>🔬 Image Diagnosis</h1>
        <p className="subtitle">
          Upload a plant leaf image to identify diseases with AI-powered analysis
        </p>
      </div>

      {/* ── Top: Upload bar ─────────────────────────────────── */}
      <div className="upload-bar">
        <div className="upload-bar-inner">
          {!preview ? (
            <div
              className="upload-dropzone-compact"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById("image-upload-input").click()}
            >
              <span className="upload-dropzone-icon">📸</span>
              <span className="upload-dropzone-text">
                Drop or click to upload a leaf image
              </span>
              <input
                id="image-upload-input"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                style={{ display: "none" }}
              />
            </div>
          ) : (
            <div className="upload-preview-row">
              <div className="upload-thumb-wrap">
                <img src={preview} alt="Uploaded leaf" className="upload-thumb" />
                <button className="btn-clear-sm" onClick={clearImage} title="Remove image">✕</button>
              </div>
              <div className="upload-info">
                <span className="upload-filename">{imageFile?.name}</span>
                {hasPrediction && (
                  <span className="upload-prediction-tag">
                    {formatClassName(result.prediction.predicted_class)}
                    <span className="tag-conf">
                      {(result.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </span>
                )}
              </div>
              <button
                id="analyze-button"
                className="btn-analyze"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? (
                  <span className="btn-loading"><span className="spinner" /> Analyzing…</span>
                ) : (
                  "🔍 Analyze"
                )}
              </button>
            </div>
          )}
          {!preview && (
            <button
              id="analyze-button"
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={!imageFile || loading}
              style={{ display: "none" }}
            >
              Analyze
            </button>
          )}
        </div>
        {error && (
          <div className="error-banner">
            <span>⚠️</span> {error}
          </div>
        )}
      </div>

      {/* ── No image yet ───────────────────────────────────── */}
      {!result && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🌿</div>
          <p>Upload an image to get started</p>
          <p className="empty-hint">
            Our AI will identify the disease, explain it, and let you ask follow-up questions
          </p>
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <div className="pulse-ring" />
          <p>Analyzing your plant image…</p>
          <p className="loading-hint">Running classification + LIME explanation</p>
        </div>
      )}

      {/* ── Results area ───────────────────────────────────── */}
      {result && (
        <div className="results-layout">
          {/* Tab switcher */}
          <div className="result-tabs">
            <button
              className={`result-tab ${activeResultTab === "analysis" ? "active" : ""}`}
              onClick={() => setActiveResultTab("analysis")}
            >
              🩺 Analysis
            </button>
            <button
              className={`result-tab ${activeResultTab === "chat" ? "active" : ""}`}
              onClick={() => setActiveResultTab("chat")}
            >
              💬 Ask Questions
              {chatMessages.length > 0 && (
                <span className="tab-badge">{chatMessages.filter(m => m.role === "user").length}</span>
              )}
            </button>
            <button
              className={`result-tab ${activeResultTab === "lime" ? "active" : ""}`}
              onClick={() => setActiveResultTab("lime")}
            >
              🔍 LIME
            </button>
          </div>

          {/* ── Analysis Tab ──────────────────────────────── */}
          {activeResultTab === "analysis" && (
            <div className="tab-content analysis-tab">
              {/* Prediction summary strip */}
              {hasPrediction && (
                <div className="prediction-strip">
                  <div className="prediction-strip-left">
                    <h2 className="prediction-name">
                      {formatClassName(result.prediction.predicted_class)}
                    </h2>
                    <div className="confidence-inline">
                      <div className="confidence-bar-mini">
                        <div
                          className="confidence-fill-mini"
                          style={{ width: `${result.prediction.confidence * 100}%` }}
                        />
                      </div>
                      <span className="confidence-pct">
                        {(result.prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <button
                    className="btn-ask-about"
                    onClick={() => setActiveResultTab("chat")}
                  >
                    💬 Ask about this
                  </button>
                </div>
              )}

              {/* Disease info sections */}
              {hasAnalysis && result.rag.status !== "generation_unavailable" && (
                <div className="disease-info">
                  {result.rag.summary && (
                    <p className="disease-summary">{result.rag.summary}</p>
                  )}
                  {result.rag.sections && result.rag.sections.length > 0 && (
                    <div className="disease-sections">
                      {result.rag.sections.map((section, i) => (
                        <details key={i} className="disease-section" open={i < 2}>
                          <summary className="disease-section-title">
                            {section.heading}
                          </summary>
                          <div className="disease-section-body">
                            {renderMarkdown(section.content_markdown)}
                          </div>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {result.rag?.status === "generation_unavailable" && (
                <div className="generation-unavailable">
                  <span>⚠️</span> {result.rag.summary}
                </div>
              )}

              {/* Sources collapsed */}
              {result.rag?.sources && result.rag.sources.length > 0 && (
                <details className="sources-collapse">
                  <summary className="sources-collapse-title">
                    📚 Sources ({result.rag.sources.length})
                  </summary>
                  <ul className="sources-list">
                    {result.rag.sources.map((src, i) => (
                      <li key={i} className="source-item">
                        <span className="source-name">{src.title}</span>
                        <span className="source-section">Section: {src.section}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}

          {/* ── Chat Tab ──────────────────────────────────── */}
          {activeResultTab === "chat" && (
            <div className="tab-content chat-tab">
              <div className="chat-container">
                <div className="chat-messages-area">
                  {/* Initial context bubble */}
                  {hasPrediction && (
                    <div className="chat-message chat-message-system">
                      <div className="chat-msg-icon">🩺</div>
                      <div className="chat-msg-content">
                        <div className="chat-msg-label">Diagnosis Context</div>
                        <p>
                          Your plant has been identified as{" "}
                          <strong>{formatClassName(result.prediction.predicted_class)}</strong>{" "}
                          with {(result.prediction.confidence * 100).toFixed(1)}% confidence.
                          Ask me anything about this condition — treatment, prevention, symptoms, and more.
                        </p>
                      </div>
                    </div>
                  )}

                  {chatMessages.length === 0 && (
                    <div className="chat-suggestions">
                      <p className="chat-suggestions-title">Try asking:</p>
                      <div className="chat-suggestion-chips">
                        {[
                          "What treatment should I use?",
                          "How can I prevent this disease?",
                          "What are the early symptoms?",
                          "Is this disease contagious to other plants?",
                        ].map((q, i) => (
                          <button
                            key={i}
                            className="chip"
                            onClick={() => {
                              setFollowUpInput(q);
                              // Auto-submit
                              setTimeout(() => {
                                document.getElementById("chat-submit-btn")?.click();
                              }, 50);
                            }}
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`chat-message chat-message-${msg.role}`}>
                      <div className="chat-msg-icon">
                        {msg.role === "user" ? "🧑" : msg.role === "error" ? "⚠️" : "🤖"}
                      </div>
                      <div className="chat-msg-content">
                        <div className="chat-msg-label">
                          {msg.role === "user" ? "You" : msg.role === "error" ? "Error" : "AI Assistant"}
                        </div>
                        <div className="chat-msg-text">
                          {msg.role === "assistant" ? renderMarkdown(msg.text) : msg.text}
                        </div>
                      </div>
                    </div>
                  ))}

                  {followUpLoading && (
                    <div className="chat-message chat-message-assistant">
                      <div className="chat-msg-icon">🤖</div>
                      <div className="chat-msg-content">
                        <div className="chat-msg-label">AI Assistant</div>
                        <div className="chat-typing">
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Chat input — always visible at bottom */}
                <form className="chat-input-bar" onSubmit={handleFollowUp}>
                  <input
                    id="followup-input"
                    type="text"
                    className="chat-input"
                    placeholder="Ask about treatment, prevention, symptoms…"
                    value={followUpInput}
                    onChange={(e) => setFollowUpInput(e.target.value)}
                    disabled={followUpLoading}
                    autoFocus
                  />
                  <button
                    id="chat-submit-btn"
                    type="submit"
                    className="btn-send"
                    disabled={!followUpInput.trim() || followUpLoading}
                  >
                    {followUpLoading ? <span className="spinner spinner-sm" /> : "Send →"}
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* ── LIME Tab ──────────────────────────────────── */}
          {activeResultTab === "lime" && (
            <div className="tab-content lime-tab">
              {result.lime ? (
                <div className="lime-content">
                  <p className="lime-description">
                    Regions highlighted show areas that influenced the prediction.
                    <strong> Green</strong> = positive contribution,
                    <strong> Red</strong> = negative contribution.
                  </p>
                  <div className="lime-images-row">
                    <div className="lime-img-card">
                      <span className="lime-img-label">Original</span>
                      <img src={preview} alt="Original" className="lime-img" />
                    </div>
                    <div className="lime-img-card">
                      <span className="lime-img-label">LIME Overlay</span>
                      <img
                        src={`data:image/png;base64,${result.lime.image}`}
                        alt="LIME explanation"
                        className="lime-img"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state-sm">
                  <p>No LIME explanation available for this image.</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
