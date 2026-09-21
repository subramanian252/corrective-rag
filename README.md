# Corrective RAG

A portfolio demonstration built from `notebooks/crag.ipynb`. It preserves the notebook's Corrective RAG workflow behind a small FastAPI application rendered with Jinja templates.

## Source and build provenance

The RAG workflow, prompts, graph stages, grading behavior, and retry behavior come from the included notebook. The FastAPI/Jinja infrastructure, repository structure, browser interface, and this documentation were built by AI from that notebook source.

The local knowledge source is the machine-learning PDF collection in `backend/data`. The PDFs are split into chunks, embedded through the OpenAI-compatible OpenRouter endpoint, and stored in the shared Pinecone index named `agenticrag`. Tavily remains the corrective source when retrieved evidence is incomplete or unsuitable.

## Workflow

1. Retrieve the six closest chunks from Pinecone.
2. Grade the retrieved documents for relevance.
3. Use accepted local evidence when it is sufficient.
4. Rewrite the question and search Tavily when local evidence needs correction.
5. Refine the combined evidence.
6. Generate the final answer from the refined context.

```text
Question -> Pinecone retrieval -> Grade documents
                                  | sufficient -> Refine -> Answer
                                  | weak -> Rewrite -> Tavily -> Refine -> Answer
```

## Shared Pinecone index

All three portfolio applications connect to `agenticrag`.

- If the index does not exist, this application creates a 1536-dimension cosine index in AWS `us-east-1`, loads every `backend/data/*.pdf` file, chunks the documents with a size of 900 and overlap of 100, and uploads the embeddings.
- If the index already exists, the saved index is opened directly and the local PDFs are not processed again.

Run this application first when you want it to create and populate the shared index.

## API

FastAPI renders the page at `GET /` and accepts questions at `POST /query`.

```http
POST /query
Content-Type: application/json

{"question": "What is machine learning?"}
```

```json
{"answer": "..."}
```

The frontend and API use the same FastAPI process. A separate frontend server is not required.

## Project structure

```text
corrective-rag/
|-- app/
|   |-- __init__.py
|   `-- main.py
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   `-- graph.py
|   |-- data/                 # Machine-learning PDF collection
|   `-- requirements.txt
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- notebooks/
|   `-- crag.ipynb
|-- .env
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.10 or newer
- OpenRouter API key
- Pinecone API key
- Tavily API key

The project-root `.env` must provide:

```dotenv
OPENROUTER_API_KEY=...
PINECONE_DB=...
TAVILY_API_KEY=...
```

The `.env` file is ignored by Git and must not be committed.

## Run locally

From the `corrective-rag` folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:api --reload
```

Open `http://127.0.0.1:8000`.

The graph is initialized when the first question is submitted. Creating and populating a new Pinecone index can make that first request take longer. Later requests reuse the initialized graph in the running process.

## Notebook-to-application mapping

| Notebook concept | Application location |
| --- | --- |
| PDF loading and chunking | `backend/app/graph.py` module setup |
| Pinecone retriever with `k=6` | `retriever` |
| Document grading | `evaluator_node` |
| Tavily correction path | `search_node` |
| Evidence refinement | `refine_node` |
| LangGraph routing | Bottom of `backend/app/graph.py` |
| Graph invocation | FastAPI `POST /query` |

## Troubleshooting

- `KeyError: OPENROUTER_API_KEY`: verify the project-root `.env` file.
- Pinecone authentication errors: verify `PINECONE_DB` and the account permissions.
- Tavily errors: verify `TAVILY_API_KEY`.
- PDF loading errors during initial index creation: confirm the files exist in `backend/data` and reinstall `pypdf`.
- Browser network errors: confirm Uvicorn is running and open `http://127.0.0.1:8000` instead of opening the HTML file directly.
