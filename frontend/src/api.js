/**
 * API service module for the Plant Disease AI backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * POST /api/rag/ask — RAG knowledge query
 */
export async function askQuestion(question) {
  const res = await fetch(`${API_BASE}/api/rag/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }

  return res.json();
}

/**
 * POST /api/diagnosis/predict — image classification + LIME
 */
export async function predictImage(imageFile) {
  const form = new FormData();
  form.append("image", imageFile);

  const res = await fetch(`${API_BASE}/api/diagnosis/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Prediction failed (${res.status})`);
  }

  return res.json();
}

/**
 * POST /api/diagnosis/analyze — combined image + question
 */
export async function analyzeImage(imageFile, question) {
  const form = new FormData();
  if (imageFile) form.append("image", imageFile);
  if (question) form.append("question", question);

  const res = await fetch(`${API_BASE}/api/diagnosis/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Analysis failed (${res.status})`);
  }

  return res.json();
}

/**
 * POST /api/diagnosis/followup — follow-up question with prediction context
 */
export async function askFollowUp(question, predictedClass, confidence) {
  const res = await fetch(`${API_BASE}/api/diagnosis/followup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      predicted_class: predictedClass,
      confidence,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Follow-up failed (${res.status})`);
  }

  return res.json();
}
