# Backend

FastAPI backend for the Enterprise HR Policy Assistant.

Module 1 adds the FastAPI application foundation, health check, settings, CORS, common schemas, and basic error handling.

## Setup

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python --version
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

RAG dependencies are intentionally separated because vector-store packages can require native build tools on Windows. Install them later when working on RAG/vector-store modules:

```powershell
pip install -r requirements-rag.txt
```

## Planned Backend Modules

- `app/api`: FastAPI routes and request/response schemas
- `app/agent`: LangGraph state, graph, nodes, prompts, and tools
- `app/rag`: document loading, chunking, embeddings, vector store, retrieval, reranking
- `app/security`: auth, permissions, PII masking, guardrails
- `app/evaluation`: golden dataset and evaluation runner
- `app/observability`: structured logs, request IDs, tracing hooks
- `app/tests`: pytest tests
- `scripts`: local indexing and maintenance scripts
- `data/raw_docs`: local sample HR documents
- `data/processed`: generated indexes or processed artifacts for local development

## Document Ingestion

Module 2 adds local document loading for:

- Markdown: `.md`, `.markdown`
- Plain text: `.txt`
- PDF: `.pdf`

Sample HR documents live in:

```text
data/raw_docs/
```

## Index Documents

Module 4 adds an indexing script:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m scripts.index_documents
```

By default, it uses mock embeddings if `OPENAI_API_KEY` is empty and stores vectors in local JSON under `VECTOR_STORE_DIR`.

For Pinecone, put these values in your local `.env`:

```text
VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=your-real-key
PINECONE_INDEX_NAME=hr-policy-assistant
PINECONE_NAMESPACE=local
```

Do not commit the real Pinecone key.

## Retrieve Context

After indexing documents, run:

```powershell
python -m scripts.retrieve_context "sick leave process" --country India --policy-type leave --top-k 3
```

This prints formatted context blocks that later chat/agent modules will pass to the LLM.

## Chat API

Module 6 adds:

```text
POST /chat
POST /api/v1/chat
```

Example:

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"What is the sick leave process?\",\"country\":\"India\",\"policy_type\":\"leave\"}"
```

The response contains:

- `answer`
- `sources`
- `used_context`
- `needs_human_confirmation`

## Agent Workflow

Module 7 routes chat through a stateful agent workflow:

```text
classify_intent -> retrieve_context -> generate_answer -> validate_response
```

The plain Python workflow works without optional packages. When `langgraph` is installed from `requirements-rag.txt`, `HRPolicyAgent.build_langgraph()` can compile the same node sequence as a LangGraph workflow.

## Environment

Create `.env` from `.env.example`. Do not commit real secrets.

## Run

```powershell
cd ai-backend-python\hr_policy_assistant\backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload
```

Health check:

```text
GET http://localhost:8000/health
```

## Test

```powershell
pytest
```
