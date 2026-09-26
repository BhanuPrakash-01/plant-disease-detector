# PRD — Intelligent Crop Disease Diagnosis & Plant Health Assistant

## 1. Project Overview

Build a web application called **Plant Disease AI Assistant**.

The system combines:

1. Deep Learning image classification
2. LIME Explainable AI
3. Retrieval-Augmented Generation using CABI plant-health content
4. Gemini LLM generation
5. React frontend
6. FastAPI backend

The final application must allow users to:

- Upload a plant-leaf image and receive a disease prediction.
- View a LIME explanation of the prediction.
- Ask questions about the predicted disease.
- Ask general plant-health questions without uploading an image.
- Receive answers grounded in the CABI knowledge base.
- View the sources used for the generated answer.

---

## 2. Existing Decisions — DO NOT CHANGE

The following components have already been developed and validated.

**Do not retrain, replace, or redesign them unless absolutely necessary for integration.**

### Image Classifier

Model:

```text
EfficientNetB0
Input: 224 × 224 × 3
Output: 29 classes
```

The classifier supports exactly these 29 classes:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy

Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy

Grape___Black_rot
Grape___Esca_(Black_Measles)
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy

Peach___Bacterial_spot
Peach___healthy

Pepper,_bell___Bacterial_spot
Pepper,_bell___healthy

Potato___Early_blight
Potato___Late_blight
Potato___healthy

Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___healthy
```

Use the exact class ordering stored with the trained model/project. Do not create a new ordering.

### Image Preprocessing

For inference:

```text
RGB
224 × 224
no training augmentation
```

Do not introduce new preprocessing or background-removal logic.

### Explainable AI

Use **LIME** with the existing classifier.

The application must be able to generate a visual explanation showing which image regions contributed positively/negatively to the predicted class.

Do not replace LIME with SHAP or another XAI method.

---

## 3. Existing RAG Knowledge Base

The RAG corpus has already been prepared from **CABI Plant Health Content**.

Current processed corpus:

```text
346 source documents
2060 usable sections
2525 final chunks
```

Chunking:

```text
~400 token maximum
~50 token overlap
```

Embedding model:

```text
BAAI/bge-m3
```

Embedding dimension:

```text
1024
```

Vector search:

```text
FAISS
IndexFlatIP
normalized embeddings
```

Existing artifacts:

```text
cabi_chunks.csv
cabi_embeddings.npy
cabi_faiss.index
```

Runtime retrieval only needs:

```text
cabi_chunks.csv
cabi_faiss.index
```

Do not regenerate embeddings or rebuild the FAISS index unless required.

The classifier remains restricted to 29 classes, but the **RAG knowledge base must remain broader than those 29 classes**.

Users must be able to ask knowledge questions about plant diseases or plant-health topics outside the classifier's 29 classes.

---

## 4. RAG Pipeline

Implement:

```text
User question
      ↓
BGE-M3
      ↓
FAISS
      ↓
Top-k relevant CABI chunks
      ↓
Gemini
      ↓
Grounded answer
      ↓
Sources
```

Default retrieval:

```text
top_k = 5
```

Retrieved chunks must contain:

```text
document_id
title
section
text
similarity score
```

---

## 5. Gemini Generation

Use the official `google-genai` Python SDK.

Configuration must come from environment variables:

```env
GEMINI_API_KEY=
GEMINI_MODEL=
```

Do not hardcode credentials.

The model name must be configurable and must not be hardcoded throughout the application.

### Generation Requirements

The LLM must:

- Use retrieved CABI information as the factual basis.
- Avoid inventing information.
- Explicitly say when the retrieved sources are insufficient.
- Preserve the distinction between classifier prediction and general plant-health knowledge.
- Never fabricate pesticide dosages or treatment instructions.
- Return relevant source information.

The backend must provide the sources separately from the generated text so the frontend does not depend on Gemini inventing citations.

---

## 6. Backend Architecture

Use **FastAPI**.

Create a clean service-oriented structure.

Recommended structure:

```text
backend/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── rag.py
│   │   └── diagnosis.py
│   │
│   ├── services/
│   │   ├── classifier.py
│   │   ├── lime_service.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   ├── schemas/
│   │   ├── rag.py
│   │   └── diagnosis.py
│   │
│   └── config.py
│
├── models/
│   └── efficientnetb0_29class.keras
│
├── rag/
│   ├── cabi_chunks.csv
│   └── cabi_faiss.index
│
├── tests/
│
├── requirements.txt
├── .env
├── .env.example
├── Dockerfile
└── README.md
```

The actual filenames may be adapted to the existing repository, but maintain the same separation of responsibilities.

---

## 7. Model Loading

Models and static artifacts must be loaded **once during application startup**, not inside individual API requests.

At startup:

```text
Load EfficientNet
Load LIME dependencies
Load BGE-M3
Load FAISS
Load CABI metadata
Initialize Gemini client
```

Requests must reuse these objects.

Do not instantiate heavy models inside endpoint handlers.

---

## 8. API Endpoints

### POST `/api/rag/ask`

Input:

```json
{
  "question": "What are the symptoms of tomato leaf miner?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "document_id": "...",
      "title": "Life cycle of the tomato leaf miner",
      "section": "Recognize the problem",
      "score": 0.6885
    }
  ]
}
```

### POST `/api/diagnosis/predict`

Input:

```text
multipart/form-data
image
```

Response should contain:

```json
{
  "prediction": "Tomato___Early_blight",
  "confidence": 0.999,
  "lime": {
    "image": "...",
    "top_positive_regions": [],
    "top_negative_regions": []
  }
}
```

The exact LIME representation can use an image URL, base64 representation, or another frontend-compatible format.

### POST `/api/diagnosis/analyze`

This is the main combined endpoint.

Input:

```text
multipart/form-data
image
question
```

Both fields should be supported.

#### When both image and question are provided

Pipeline:

```text
Image
 ↓
EfficientNet
 ↓
Prediction
 ↓
LIME

Question
 ↓
BGE-M3
 ↓
FAISS
 ↓
CABI context

Prediction + retrieved context + user question
 ↓
Gemini
 ↓
Final response
```

Response:

```json
{
  "prediction": {
    "class": "Tomato___Early_blight",
    "confidence": 0.999
  },
  "lime": {},
  "answer": "...",
  "sources": []
}
```

#### Image only

Return:

```text
prediction
confidence
LIME explanation
```

#### Question only

Handle as a normal RAG query.

---

## 9. Frontend

Build using:

```text
React
Vite
```

Use a clean, modern, responsive interface.

Do not over-design the UI.

The application should have two primary interaction paths.

### Image Diagnosis

Components:

```text
Upload image
Preview image
Analyze button
Prediction
Confidence
LIME explanation
Question box
Answer
Sources
```

### Plant Health Knowledge

Components:

```text
Question input
Ask button
Answer
Sources
```

The user must be able to use the knowledge assistant **without uploading an image**.

---

## 10. Image Diagnosis UI

After prediction display:

```text
Disease:
Tomato Early Blight

Confidence:
99.9%
```

Display the LIME explanation visually.

The UI must clearly distinguish:

```text
Prediction = model classification

Explanation = LIME interpretation of that prediction

Knowledge answer = retrieved CABI information + Gemini
```

Do not present the generated answer as if it were the classifier's prediction.

---

## 11. RAG Source Display

Every generated answer should display source information.

Example:

```text
Sources

Life cycle of the tomato leaf miner
Section: Recognize the problem

Strategies for Sustainable Management of the Tomato Leafminer
Section: Introduction
```

Do not allow the frontend to invent source information.

Use metadata returned by the backend.

---

## 12. Error Handling

Implement handling for:

```text
Invalid image
Unsupported image format
Missing question
Empty question
Model loading failure
FAISS loading failure
Gemini API failure
Gemini timeout
RAG retrieval failure
LIME failure
```

The API must return clear HTTP status codes and readable error messages.

Do not expose API keys, stack traces, filesystem paths, or internal secrets to users.

---

## 13. Performance Requirements

The application must avoid unnecessary work.

The following must happen only once per process:

```text
BGE-M3 loading
EfficientNet loading
FAISS loading
CABI metadata loading
```

Do not generate embeddings for the entire knowledge base at runtime.

Only the user query is embedded at runtime.

FAISS should retrieve directly from the existing index.

Do not download models on every request.

---

## 14. Testing Requirements

Create automated tests for the following.

### RAG

```text
query → retrieval
query → generation
source metadata
empty retrieval
Gemini failure
```

### Classifier

```text
valid image
invalid image
prediction format
confidence
29-class output
```

### LIME

```text
explanation generation
invalid image handling
```

### API

Test all three endpoints:

```text
/api/rag/ask
/api/diagnosis/predict
/api/diagnosis/analyze
```

### Integration

Test:

```text
image only
question only
image + question
```

---

## 15. Environment Configuration

Create:

```text
.env.example
```

Containing:

```env
GEMINI_API_KEY=
GEMINI_MODEL=

EMBEDDING_MODEL=BAAI/bge-m3

RAG_TOP_K=5
```

No secrets in source code.

`.gitignore` must exclude:

```text
.venv/
.env
__pycache__/
*.pyc
.DS_Store
```

---

## 16. Dependency Management

Create a proper `requirements.txt` based on the working local environment.

Use compatible, reproducible package versions.

Do not install unnecessary libraries.

The application must run inside the project's `.venv`.

---

## 17. Docker

Create a production-ready Dockerfile for the FastAPI backend.

Requirements:

- Start FastAPI with a production server.
- Load models at startup.
- Do not download models for every request.
- Do not require Google Colab.
- Do not depend on Google Drive.
- Required runtime artifacts must be packaged with the application or loaded from a configured artifact/model registry.
- Container build must be reproducible.

---

## 18. Model and Artifact Storage

Prepare the application so these artifacts can be stored/versioned externally:

```text
EfficientNet model
BGE-M3 model
FAISS index
CABI metadata
```

Hugging Face may be used as the artifact/model registry.

The application must not download the BGE-M3 model for every request.

---

## 19. Deployment Readiness

Do not deploy immediately.

First ensure:

```text
Backend works locally
Frontend works locally
All APIs work
RAG works
Classifier works
LIME works
Combined workflow works
Docker build succeeds
Production environment variables are documented
```

Prepare the project so it can later be deployed as:

```text
Frontend → Vercel or equivalent
Backend → Google Cloud Run or equivalent
```

The deployment provider must not be hardcoded into application logic.

---

## 20. Important Constraints

Do **not**:

- Retrain EfficientNet.
- Change the 29-class scope.
- Replace LIME.
- Rebuild the CABI dataset.
- Replace BGE-M3.
- Rebuild FAISS unnecessarily.
- Add a database unless there is a genuine application requirement.
- Put Gemini API keys in code.
- Put `.env` in Git.
- Put model initialization inside request handlers.
- Add LangChain, LangGraph, Celery, Kafka, Redis, Kubernetes, or other infrastructure unless genuinely required for an identified application problem.

Keep the initial system simple and modular.

---

## 21. Final Application Flows

### Scenario A — Knowledge Query

```text
User question
      ↓
BGE-M3
      ↓
FAISS
      ↓
Top CABI chunks
      ↓
Gemini
      ↓
Answer + sources
```

### Scenario B — Image Diagnosis

```text
Leaf image
      ↓
EfficientNetB0
      ↓
29-class prediction
      ↓
LIME explanation
      ↓
User question
      ↓
BGE-M3
      ↓
FAISS
      ↓
CABI context
      ↓
Gemini
      ↓
Prediction + explanation + grounded answer + sources
```

---

## 22. Definition of Done

The implementation is complete when:

```text
✓ React frontend runs locally
✓ FastAPI backend runs locally
✓ Image upload works
✓ EfficientNet prediction works
✓ Confidence is returned
✓ LIME explanation is displayed
✓ RAG questions work without an image
✓ Image + question workflow works
✓ CABI sources are displayed
✓ Gemini generation works
✓ Errors are handled
✓ Tests pass
✓ Docker image builds successfully
✓ README contains setup/run instructions
✓ No secrets are committed
✓ Application does not depend on Google Colab
✓ Application is ready for deployment
```

---

# Instruction to Antigravity

**Implement this PRD against the existing project. Before changing anything, inspect the repository and identify existing model files, RAG artifacts, class mappings, preprocessing code, and LIME implementation. Reuse validated existing components wherever possible. Do not replace working ML/RAG components with new implementations without a concrete compatibility reason. Build the backend first, then the frontend, then integration tests, then Docker/deployment readiness. At each major stage, run the application/tests and fix integration issues before proceeding. Do not deploy yet.**
