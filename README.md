# OpsRecall (React + RAG)

This repo contains a DevOps RAG assistant:
- `rag_core.py` holds the RAG logic (Chroma retrieval + LLM answer).
- `backend/api_server.py` exposes a small HTTP API for the UI: `POST /api/ask`.
- `frontend/` is a React chat UI that calls the backend API.

## 1) Prerequisites

1. Ensure you have `OPENROUTER_API_KEY` in the root `.env` file.
2. Your Chroma DB must exist in `chroma_db/` (created by your existing scripts).

## 2) Start the backend API

Run:

```powershell
python backend/api_server.py
```

By default the API listens on `http://localhost:8000`.

Optional environment variables:
- `RAG_API_PORT` (default: `8000`)
- `RAG_API_HOST` (default: `0.0.0.0`)

## 3) Start the React UI

Run:

```powershell
cd frontend
npm run dev
```

Open the URL shown by Vite (typically `http://localhost:5173`).

The UI calls:
- `http://localhost:8000/api/ask` via a Vite dev proxy (`/api/ask`)

## 4) Debug / retrieval settings

In the UI you can control:
- `Retrieved chunks` (maps to `top_k`)
- `Debug` (when enabled, the UI can expand retrieved chunks and the prompt)

