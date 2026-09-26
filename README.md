# Plant Disease AI Assistant

An intelligent crop disease diagnosis and plant health assistant that combines deep learning image classification, LIME explainable AI, and RAG-powered knowledge retrieval.

## Features

- **Image Diagnosis** — Upload a plant-leaf image and receive a disease prediction from an EfficientNetB0 classifier (29 classes)
- **LIME Explainability** — See which image regions influenced the prediction
- **Plant Health Knowledge** — Ask questions about any plant disease, powered by the CABI knowledge base + Gemini LLM
- **Combined Workflow** — Upload an image and ask a follow-up question in one step

## Tech Stack

| Component        | Technology                     |
|-----------------|-------------------------------|
| Frontend        | React + Vite                  |
| Backend         | FastAPI (Python)              |
| Image Classifier| EfficientNetB0 (Keras/TF)     |
| Explainability  | LIME                          |
| Embeddings      | BAAI/bge-m3                   |
| Vector Search   | FAISS (IndexFlatIP)           |
| Knowledge Base  | CABI Plant Health (2525 chunks)|
| LLM             | Google Gemini                 |

## Prerequisites

- Python 3.12+
- Node.js 18+
- A [Google Gemini API key](https://aistudio.google.com/)

## Quick Start

### 1. Clone and set up backend

```bash
cd backend

# Create virtual environment
python -m venv ../.venv
source ../.venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Start the backend

```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The backend will load all models at startup (~30-60 seconds for first load). You'll see:
```
═══ All models loaded — ready to serve ═══
```

### 3. Set up and start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ask` | Ask a plant-health question (RAG) |
| `POST` | `/api/diagnosis/predict` | Upload image → prediction + LIME |
| `POST` | `/api/diagnosis/analyze` | Image and/or question → combined |
| `GET`  | `/health` | Health check |

### Example: RAG Query

```bash
curl -X POST http://localhost:8000/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of tomato early blight?"}'
```

### Example: Image Prediction

```bash
curl -X POST http://localhost:8000/api/diagnosis/predict \
  -F "image=@leaf.jpg"
```

## Running Tests

```bash
cd backend
source ../.venv/bin/activate
python -m pytest tests/ -v
```

## Docker

```bash
cd backend
docker build -t plant-disease-ai .
docker run -p 8000:8000 --env-file .env plant-disease-ai
```

## Project Structure

```
plant-disease-ai/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI route handlers
│   │   │   ├── diagnosis.py
│   │   │   └── rag.py
│   │   ├── schemas/       # Pydantic request/response models
│   │   │   ├── diagnosis.py
│   │   │   └── rag.py
│   │   ├── services/      # Business logic
│   │   │   ├── classifier.py
│   │   │   ├── generator.py
│   │   │   ├── lime_service.py
│   │   │   └── retriever.py
│   │   ├── config.py
│   │   └── main.py
│   ├── models/            # EfficientNetB0 model
│   ├── rag/               # FAISS index + CABI chunks
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── api.js
    │   ├── App.jsx
    │   └── main.jsx
    ├── index.html
    └── package.json
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | (required) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Sentence transformer model | `BAAI/bge-m3` |
| `RAG_TOP_K` | Number of CABI chunks to retrieve | `5` |

## Deployment

Prepared for deployment as:
- **Frontend** → Vercel or equivalent static hosting
- **Backend** → Google Cloud Run or equivalent container hosting

No deployment provider is hardcoded into the application.

## Constraints

- The image classifier is restricted to **29 classes** (PlantVillage subset)
- The RAG knowledge base covers **broader** plant-health topics beyond those 29 classes
- LIME is used for explainability (not SHAP)
- No database is required — all data is loaded from files at startup
