# Enterprise HR Policy Assistant

Enterprise-style Agent + RAG project for learning and interview preparation. The assistant will answer HR policy questions using approved company documents, show citations, and later support agent tools such as email drafting, employee profile lookup, approval flows, and HR ticket creation.

This repository is intentionally built module by module. Module 0 creates only the scaffold and setup instructions. Real RAG, agent workflow, APIs, tests, and UI features will be added in later modules.

## Tech Stack

MVP stack:

- Backend: FastAPI, Uvicorn, Pydantic, pydantic-settings
- Agent workflow: LangGraph
- RAG: LangChain or lightweight custom pipeline
- Vector store: Chroma or FAISS
- LLM/embeddings: OpenAI or Azure OpenAI
- UI: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui, lucide-react
- Testing: pytest

Enterprise upgrade path:

- PostgreSQL + pgvector
- Redis + Celery
- OAuth2/JWT or SSO
- OpenTelemetry, Prometheus, Grafana
- LangSmith or Arize Phoenix
- Docker, CI/CD, cloud storage, secrets manager

## Project Structure

```text
hr_policy_assistant/
  backend/
    app/
      api/
      agent/
      rag/
      security/
      evaluation/
      observability/
      tests/
      main.py
      config.py
    scripts/
    data/
      raw_docs/
      processed/
    requirements.txt
    .env.example
    README.md
  frontend/
    app/
    components/
    lib/
    public/
    README.md
  docs/
    MODULE_PROGRESS.md
  README.md
```

## Backend Setup

From the project root:

```powershell
cd ai-backend-python\hr_policy_assistant\backend
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Run the backend:

```powershell
uvicorn app.main:app --reload
```

Health check:

```text
GET http://localhost:8000/health
```

## Frontend Setup

The frontend folder is scaffolded now. A real Next.js app will be created in the frontend modules.

Expected stack:

```text
Next.js + React + TypeScript + Tailwind CSS + shadcn/ui
```

## Module Status

- Module 0: Project Scaffold And Setup - completed
- Module 1: Backend Foundation - completed
- Module 2: Document Ingestion - completed
- Module 3: Chunking And Metadata - completed
- Module 4: Embeddings And Vector Store - completed
- Module 5: Basic RAG Retriever - completed
- Module 6: Chat API With Citations - completed
- Module 7: Agent Workflow With LangGraph - completed
- Module 8: Tools And Approval Flow - next

## Next Step

Ask:

```text
AGENT_RAG_PROJECT_MEMORY.md read karo.
Ab Module 8: Tools And Approval Flow implement karo.
Previous modules ko break mat karna.
Tests run karo aur final me kya bana uska short summary do.
```
