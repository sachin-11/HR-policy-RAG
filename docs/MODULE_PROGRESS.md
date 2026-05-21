# Module Progress

## Module 0: Project Scaffold And Setup

Status: Completed

Created:

- root `hr_policy_assistant` project folder
- backend scaffold
- frontend scaffold
- backend `requirements.txt`
- backend `.env.example`
- setup instructions
- placeholder packages for future modules

No real RAG, API behavior, agent workflow, tests, or UI implementation has been added yet.

## Next Module

Module 1: Backend Foundation

Status: Completed

Created:

- FastAPI app factory in `backend/app/main.py`
- `GET /health`
- root `GET /`
- pydantic-settings configuration in `backend/app/config.py`
- common schemas in `backend/app/api/schemas.py`
- CORS setup
- standard `AppError` response handling
- validation and unhandled exception handlers
- basic pytest coverage
- minimal Module 1 requirements separated from later RAG/vector-store dependencies

Run:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
uvicorn app.main:app --reload
```

Test:

```powershell
pytest
```

Verification note:

Tests are present, but they could not be executed in the current shell because `python`, `py`, and `pytest` are not available on PATH. After Python is installed and dependencies are installed, run `pytest` from `backend/`.

Dependency note:

`chromadb` and other RAG dependencies are moved to `backend/requirements-rag.txt` for later modules. Module 1 only needs `backend/requirements.txt`.

## Next Module

Module 2: Document Ingestion

Status: Completed

Created:

- `backend/app/rag/loaders.py`
- `LoadedDocument` and `DocumentMetadata` schemas
- Markdown/text/PDF loader support
- simple Markdown front matter metadata extraction
- source metadata helper for future citations
- sample HR documents in `backend/data/raw_docs`
- loader tests in `backend/app/tests/test_loaders.py`

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

Verification note:

Tests are present, but they could not be executed from the assistant shell because `python` is not available on PATH there. Run `python -m pytest` from your activated backend environment.

## Next Module

Module 3: Chunking And Metadata

Status: Completed

Created:

- `backend/app/rag/chunking.py`
- `DocumentChunk` and `ChunkMetadata` schemas
- section-aware Markdown heading splitting
- character fallback chunking with overlap
- stable chunk IDs
- metadata preservation from loaded documents
- vector-store friendly record helper
- chunking tests in `backend/app/tests/test_chunking.py`
- sample output notes in `backend/data/processed/CHUNKING_NOTES.md`

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

## Next Module

Module 4: Embeddings And Vector Store

Status: Completed

Created:

- `backend/app/rag/embeddings.py`
- `EmbeddingProvider` abstraction
- deterministic `MockEmbeddingProvider` for tests/offline development
- `OpenAIEmbeddingProvider` with lazy OpenAI import
- `backend/app/rag/vector_store.py`
- `VectorStore` abstraction
- local JSON vector store for tests/MVP
- optional Pinecone vector store wrapper
- `backend/scripts/index_documents.py`
- mock embedding and vector-store tests
- Pinecone environment placeholders in `.env.example`

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

Verification note:

Tests are present, but the assistant shell could not execute them because the existing `.venv` points to a missing Python executable. Recreate the venv if you see `No Python at ...Python312...`.

Index locally:

```powershell
python -m scripts.index_documents
```

Pinecone note:

Set `VECTOR_STORE_PROVIDER=pinecone` and `PINECONE_API_KEY` in local `.env` only. Never commit real keys.

## Next Module

Module 5: Basic RAG Retriever

Status: Completed

Created:

- `backend/app/rag/retriever.py`
- `RagRetriever`
- `RetrievalFilters`
- `RetrievedChunk` and `RetrievalResponse`
- top-k search orchestration
- metadata filter normalization
- score threshold filtering
- formatted context builder
- compact source citation builder
- example retrieval command in `backend/scripts/retrieve_context.py`
- retriever tests in `backend/app/tests/test_retriever.py`

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

Example:

```powershell
python -m scripts.index_documents
python -m scripts.retrieve_context "sick leave process" --country India --policy-type leave --top-k 3
```

## Next Module

Module 6: Chat API With Citations

Status: Completed

Created:

- `backend/app/api/chat_routes.py`
- `POST /chat`
- `POST /api/v1/chat`
- chat request/response schemas
- `backend/app/agent/prompts.py`
- RAG prompt template
- `backend/app/agent/llm.py`
- LLM client abstraction
- local extractive fallback LLM client
- optional OpenAI chat client
- answer with citations
- no-context fallback
- chat API tests with fake retriever and fake LLM

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

Example:

```powershell
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"What is the sick leave process?\",\"country\":\"India\",\"policy_type\":\"leave\"}"
```

## Next Module

Module 7: Agent Workflow With LangGraph

Status: Completed

Created:

- `backend/app/agent/state.py`
- `AgentState` shared workflow state
- `backend/app/agent/nodes.py`
- intent classification node
- retrieval node
- answer generation node
- response validation node
- `backend/app/agent/graph.py`
- `HRPolicyAgent` plain Python stateful runner
- optional LangGraph compiled workflow builder
- chat API now uses the agent workflow internally
- workflow tests in `backend/app/tests/test_agent_workflow.py`

Test:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m pytest
```

Optional LangGraph:

```powershell
python -m pip install -r requirements-rag.txt
```

## Next Module

Module 8: Tools And Approval Flow

Expected work:

- employee profile tool
- email draft tool
- HR ticket draft/create tool
- approval-required response
- direct execution block for sensitive tools
