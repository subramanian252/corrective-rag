# Corrective RAG

A small portfolio application built from the original `crag.ipynb` notebook. The project keeps the notebook's Corrective RAG flow and places a minimal FastAPI boundary and a focused web interface around it.

The interface follows the visual language of the supplied portfolio reference: warm paper tones, dark green typography, editorial spacing, hand-drawn details, and an illustrated landscape. The product surface stays intentionally small—one question form and one answer.

## Source and build provenance

All RAG behavior, prompts, graph steps, thresholds, retrieval settings, and source information in this project are sourced from the included `notebooks/crag.ipynb` notebook. The application infrastructure—including the FastAPI/Jinja integration, project structure, browser interface, and documentation—was built by AI from that notebook source.

The local RAG source is `backend/data/book.pdf`: **Hands-On Machine Learning with Scikit-Learn and TensorFlow** by Aurélien Géron. The notebook also uses Tavily web search as a corrective source when the local passages are graded partial or incorrect.

## What it does

The graph retrieves passages from the included machine-learning PDF, evaluates the relevance of every retrieved document, and chooses one of two paths:

1. If at least one document scores above `0.8`, use the accepted local passages.
2. If the local evidence is partial or incorrect, rewrite the question and search the web with Tavily.
3. Split the combined evidence into sentences and keep the useful sentences.
4. Generate an answer using only the refined context.

This is the same architecture used in the notebook. The application code only adapts paths, serialization, and the HTTP boundary required to run it as a web project.

## Architecture

```text
Browser
  │ POST /query
  ▼
FastAPI
  ▼
Retrieve from FAISS
  ▼
Grade documents ── correct ─────────────┐
  │                                      │
  └── partial / incorrect → Tavily search│
                                         ▼
                              Refine sentence context
                                         ▼
                                   Generate answer
```

FastAPI serves the Jinja page at `GET /` and accepts questions at `POST /query`:

```http
POST /query
Content-Type: application/json

{"question": "What is machine learning?"}
```

Response:

```json
{"answer": "..."}
```

FastAPI's automatic documentation URLs are disabled. The frontend and API are served by the same process.

## Project structure

```text
corrective-rag/
├── app/
│   ├── __init__.py
│   └── main.py            # FastAPI + Jinja application entry point
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── graph.py       # Notebook graph adapted for application use
│   ├── data/
│   │   └── book.pdf       # Original local knowledge source
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── notebooks/
│   └── crag.ipynb         # Unmodified source notebook
├── .env                   # Copied locally; ignored by Git
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenRouter API key
- A Tavily API key

The copied `.env` file is loaded from the project root. It must provide:

```dotenv
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

Do not commit `.env`; it is already covered by `.gitignore`.

## Run locally

From the project root, create an environment, install the dependencies, and start the single FastAPI process:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:api --reload
```

Open `http://127.0.0.1:8000`. FastAPI renders `frontend/index.html` through `Jinja2Templates`, serves the CSS and JavaScript, and handles `/query`; a separate frontend server is not required.

The web page starts immediately. On the first question, the backend loads `backend/data/book.pdf`, divides it into 900-character chunks with 100-character overlap, creates OpenAI-compatible embeddings through OpenRouter, and builds an in-memory FAISS index. That initialized graph is reused for later questions in the same process.

## Notebook-to-app mapping

| Notebook element | Application location |
| --- | --- |
| PDF loading and chunking | `backend/app/graph.py` module setup |
| FAISS retriever with `k=3` | `backend/app/graph.py` |
| Document evaluator and thresholds | `evaluator_node` |
| Tavily query rewrite and search | `search_node` |
| Sentence filtering | `refine_node` |
| LangGraph edges and routing | bottom of `graph.py` |
| `app.invoke(...)` | FastAPI `POST /query` |

## Notes

- The FAISS index is built in memory whenever the backend process starts; this mirrors the notebook and avoids adding persistence not present in the original project.
- A partial local result follows the notebook's web-search correction path.
- The frontend calls `/query` on the same FastAPI origin, so no CORS or second port is needed.
- The response time depends on embedding, grading, search, filtering, and generation calls. A question that needs web correction will take longer than one answered by local evidence.

## Troubleshooting

- `KeyError: OPENROUTER_API_KEY`: confirm the project-root `.env` contains that variable.
- Tavily authentication errors: confirm `TAVILY_API_KEY` is present and active.
- Browser network error: confirm Uvicorn is running and open `http://127.0.0.1:8000` rather than opening the HTML file directly.
- PDF or FAISS startup errors: reinstall `pypdf` and `faiss-cpu` from `backend/requirements.txt`.
