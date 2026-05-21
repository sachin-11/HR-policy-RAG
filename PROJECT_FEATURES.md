# HR Policy Assistant — Complete Feature Reference

> **Purpose:** Yeh file future conversations ke liye memory ka kaam karegi.
> Is file ko share karo taaki kisi bhi conversation mein project ka poora context mil sake.

---

## Project Overview

**Enterprise HR Policy Assistant** — ek AI-powered chatbot jo employees ke HR policy questions answer karta hai. RAG (Retrieval Augmented Generation) architecture use karta hai jisme real HR documents se context fetch hota hai aur OpenAI GPT se answer generate hota hai.

**Stack:**
- **Backend:** Python 3.12, FastAPI, LangGraph, OpenAI, Pinecone / Local JSON vector store
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS, react-markdown
- **Auth:** Custom HMAC-based JWT (no external library)
- **Email:** SMTP (Gmail / Google Workspace)
- **Vector Store:** Local JSON (dev) + Pinecone (prod)

---

## Project Structure

```
hr_policy_assistant/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   │   ├── auth_routes.py        # Admin token generation
│   │   │   ├── chat_routes.py        # Chat + Streaming endpoints
│   │   │   ├── document_routes.py    # Document upload/indexing
│   │   │   ├── session_routes.py     # Conversation session management
│   │   │   └── schemas.py            # Shared Pydantic schemas
│   │   ├── agent/            # LangGraph AI agent
│   │   │   ├── graph.py              # HRPolicyAgent + LangGraph workflow
│   │   │   ├── nodes.py              # Workflow nodes
│   │   │   ├── state.py              # AgentState TypedDict
│   │   │   ├── prompts.py            # RAG prompt templates
│   │   │   ├── llm.py                # LLM clients (OpenAI + fallback)
│   │   │   ├── tools.py              # Agent tools (email, ticket, profile)
│   │   │   └── email_compose.py      # Email draft composer
│   │   ├── rag/              # Retrieval pipeline
│   │   │   ├── loaders.py            # MD/TXT/PDF document loaders
│   │   │   ├── chunking.py           # Section-aware chunking
│   │   │   ├── embeddings.py         # OpenAI + Mock embedding providers
│   │   │   ├── vector_store.py       # Local JSON + Pinecone vector stores
│   │   │   ├── retriever.py          # RAG retriever with filters
│   │   │   └── indexing.py           # Full index pipeline runner
│   │   ├── security/         # Auth & safety
│   │   │   ├── auth.py               # JWT create/decode, role guards
│   │   │   ├── permissions.py        # RBAC + authorize_chat_request
│   │   │   └── pii.py                # PII masking + prompt injection guard
│   │   ├── sessions/         # Conversation history
│   │   │   └── store.py              # In-memory SessionStore singleton
│   │   ├── mail/             # SMTP email
│   │   │   └── smtp_client.py        # send_plain_text_email
│   │   ├── observability/    # Logging & tracing
│   │   │   ├── logging.py            # Structured JSON logging (log_event)
│   │   │   └── middleware.py         # Request ID + latency middleware
│   │   ├── evaluation/       # Quality evaluation
│   │   ├── tests/            # pytest test suite
│   │   ├── config.py         # Pydantic Settings (env vars)
│   │   └── main.py           # FastAPI app factory
│   ├── scripts/
│   │   ├── index_documents.py        # CLI: chunk + embed + upsert all docs
│   │   └── retrieve_context.py       # CLI: test retrieval
│   └── data/
│       ├── raw_docs/                 # Source HR policy documents
│       └── processed/                # Vector store + chunking notes
├── frontend/
│   └── app/
│       ├── page.tsx                  # Main chat UI
│       └── admin/documents/page.tsx  # Admin panel
└── PROJECT_FEATURES.md               # Yeh file
```

---

## Backend — Sab Features

### 1. FastAPI Foundation
- `GET /health` — health check (status, version, env)
- `GET /` — root message
- CORS configured for frontend origin
- Standard `AppError` + `ErrorResponse` JSON format
- Request validation error handler (422)
- Unhandled exception handler (500, debug mode mein details)

### 2. Document Ingestion (`app/rag/loaders.py`)
- **Supported formats:** `.md`, `.markdown`, `.txt`, `.pdf`
- Markdown front matter metadata extraction (title, country, policy_type, etc.)
- `LoadedDocument` + `DocumentMetadata` Pydantic schemas
- Source metadata helper for citations

### 3. Chunking (`app/rag/chunking.py`)
- Section-aware Markdown heading splitting (`#`, `##`, `###`)
- Character fallback chunking with overlap
- Stable chunk IDs (hash-based)
- Metadata preserved from parent document
- Vector-store friendly record helper

### 4. Embeddings & Vector Store
- **Embedding providers:** `OpenAIEmbeddingProvider` (text-embedding-3-small) + `MockEmbeddingProvider` (offline/test)
- **Vector stores:** `LocalJsonVectorStore` (dev, no internet needed) + `PineconeVectorStore` (prod)
- Auto-detect mock dimension from existing local store
- `build_embedding_provider()` + `build_vector_store()` factory functions

### 5. RAG Retriever (`app/rag/retriever.py`)
- Top-K similarity search
- Metadata filters: `country`, `employee_type`, `policy_type`, `access_level`
- Score threshold filtering
- `format_retrieved_context()` — formatted context blocks for LLM
- `build_source_citations()` — citation objects for frontend
- `dedupe_source_citations_for_display()` — remove duplicate sources

### 6. LangGraph Agent Workflow (`app/agent/`)

**Agent flow:**
```
classify_intent
    ├── action_request → execute_tools → retrieve_context → generate_answer → validate_response
    └── policy_qa / general_hr / unknown → retrieve_context → generate_answer → validate_response
```

**Nodes:**
| Node | Kaam |
|------|------|
| `classify_intent_node` | Keywords se intent detect karo (policy_qa / action_request / general_hr) |
| `execute_tools_node` | Safe tools chalao, sensitive tools ke liye approval block karo |
| `retrieve_context_node` | RAG se relevant chunks fetch karo |
| `generate_answer_node` | LLM se answer generate karo (conversation history + context) |
| `validate_response_node` | Empty answer check, approval flag set, tool output merge |

**AgentState fields:**
- `user_message`, `user_id`, `intent`, `filters`, `top_k`, `score_threshold`
- `conversation_history` — previous turns ka formatted block
- `retrieved_chunks`, `context`, `used_context`
- `tool_results`, `approval_required_actions`, `sources`, `errors`
- `prompt`, `final_answer`, `needs_human_confirmation`

**LangGraph features (jab `langgraph` install ho):**
- `MemorySaver` checkpointing per `thread_id`
- Conditional routing (action_request → tools, baaki → retrieve)
- `interrupt()` for human-in-the-loop approval
- `astream()` for per-node streaming

### 7. Agent Tools (`app/agent/tools.py`)

| Tool | Actions | Approval Required |
|------|---------|------------------|
| `employee_profile` | `lookup` | No |
| `email_draft` | `draft` | No |
| `email_draft` | `send` | **Yes** |
| `hr_ticket` | `draft` | No |
| `hr_ticket` | `create` | **Yes** |

- `EmployeeProfileTool` — mock employee profile (country, manager, employee_type)
- `EmailDraftTool` — compose + send via SMTP (manager_leave / hr_sick_leave)
- `HRTicketTool` — draft / create HR support tickets
- `ToolOrchestrator` — plan karo, safe tools chalao, blocked ones approval queue mein daalo

### 8. LLM Clients (`app/agent/llm.py`)

| Client | Kaam |
|--------|------|
| `OpenAIChatClient` | GPT-4o-mini (ya koi bhi model), temperature=0.2 |
| `ExtractiveLLMClient` | Offline fallback, context se extract karta hai |

- `generate(prompt)` — synchronous response
- `generate_freeform(prompt)` — email/freeform generation
- `stream_generate(prompt)` — **async generator**, token-by-token streaming
  - `OpenAIChatClient`: AsyncOpenAI real streaming
  - `ExtractiveLLMClient`: full text ek shot mein yield

### 9. Chat API (`app/api/chat_routes.py`)

**`POST /chat`** (aur `/api/v1/chat`)
- Request: `message`, `session_id?`, `country?`, `employee_type?`, `access_level?`, `department?`, `policy_type?`, `top_k`, `score_threshold?`, `approved_tool_actions?`
- Response: `answer`, `sources`, `used_context`, `needs_human_confirmation`, `intent`, `tool_results`, `approval_required_actions`, `session_id`

**`POST /chat/stream`** (aur `/api/v1/chat/stream`)
- Server-Sent Events (SSE), `text/event-stream`
- Events:
  ```
  data: {"type": "token", "text": "..."}     ← har token
  data: {"type": "done", "sources": [...], "session_id": "..."}  ← end
  data: {"type": "error", "message": "..."}  ← error case
  ```
- Same auth/PII/session logic as non-streaming endpoint

### 10. Security (`app/security/`)

**Authentication (`auth.py`):**
- Custom HMAC-SHA256 JWT (no PyJWT dependency)
- Roles: `anonymous`, `employee`, `manager`, `hr`, `admin`
- `create_access_token(data, expires_in?)` — token banana
- `decode_access_token(token)` — validate + parse
- `get_current_user()` — FastAPI dependency (anonymous fallback)
- `get_admin_user()` — admin-only guard

**Permissions (`permissions.py`):**
- `UserContext` — requesting user ka profile
- `authorize_chat_request()` — filters sanitize karo (employee apna hi data dekh sakta hai)
- `PermissionDenied` exception

**PII & Safety (`pii.py`):**
- Email, phone, Aadhaar-style ID, credit card masking
- `mask_pii(text)` → `[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, etc.
- `contains_prompt_injection(text)` — "ignore previous instructions" jaise phrases detect karo

### 11. Admin Token Generation (`app/api/auth_routes.py`)

**`POST /auth/admin-token`** (aur `/api/v1/auth/admin-token`)
- Request: `{"password": "..."}`
- Response: `{"token": "...", "expires_in": 86400}`
- Password: `.env` mein `ADMIN_PASSWORD` (default: `admin123`)
- HMAC compare_digest se secure comparison

### 12. Document Management API (`app/api/document_routes.py`)

Sab routes admin JWT required:

| Method | Route | Kaam |
|--------|-------|------|
| `GET` | `/documents` | Raw docs folder ki file list |
| `GET` | `/documents/indexing-status` | Last indexing job ka status + live vector count |
| `POST` | `/documents/upload` | File upload (max 10 MB, `.md/.txt/.pdf`) |
| `POST` | `/documents/reindex` | Background mein chunk + embed + upsert |

- Background task se indexing (`threading.Lock` se concurrency guard)
- Status file `data/processed/indexing_status.json` mein persist hoti hai
- Auto-polling frontend ke liye job_state track karta hai

### 13. Conversation History / Sessions (`app/sessions/store.py`)

- `SessionStore` — process-wide singleton, thread-safe `dict`
- `ConversationSession` — session_id, user_id, messages list, created_at, updated_at
- `ConversationMessage` — role (user/assistant), content, timestamp
- TTL: **2 hours** inactivity ke baad auto-expire
- Max history: **10 turns** (20 messages) per session
- `to_prompt_block()` — history ko LLM-ready format mein convert

**`GET /sessions/{session_id}`** — conversation history fetch
**`DELETE /sessions/{session_id}`** — session clear karo

### 14. Email (`app/mail/smtp_client.py`)
- `send_plain_text_email(to, subject, body)`
- Gmail / Google Workspace SMTP (port 587, STARTTLS)
- `smtp_is_configured()` — SMTP_USER + SMTP_PASS dono hone chahiye

### 15. Observability (`app/observability/`)
- `log_event(logger, event, **kwargs)` — structured JSON logs
- `RequestIDMiddleware` — har request ko unique ID deta hai
- Latency logging per request
- `LOG_LEVEL` + `ENABLE_TRACING` env vars

---

## Frontend — Sab Features

### Main Chat Page (`app/page.tsx`)

**Layout:**
- Fixed header (sticky, scrolling se nahi jaata) — `h-screen` layout
- Scrollable messages area (`flex-1 overflow-y-auto`)
- Fixed input bar at bottom

**Chat Features:**
1. **Streaming Responses** — token-by-token live text display
   - "Thinking..." tab tak dikhe jab tak pehla token na aaye
   - Blinking teal cursor `▌` streaming ke dauran
   - `fetch` + `ReadableStream` + SSE parser
2. **Markdown Rendering** — `react-markdown` + `remark-gfm`
   - Tables, bold, lists, code, headings sab properly render hote hain
   - `@tailwindcss/typography` prose classes se styled
3. **Suggested Questions** — welcome message ke neeche 4 quick-prompt pills
   - "What is the sick leave policy?"
   - "How do I apply for maternity leave?"
   - "What are the WFH guidelines?"
   - "How to claim medical reimbursement?"
4. **Conversation History** — `session_id` har request ke saath bheja jaata hai multi-turn memory ke liye
5. **Human Approval Modal** — sensitive tool actions (email send, ticket create) ke liye confirmation dialog
6. **Source Citations** — har answer ke saath document sources dikhte hain (pill badges)
7. **Feedback Buttons** — Thumbs up/down (Yes/No) for last answer
8. **PII + Injection** — backend automatically mask + block karta hai

### Admin Panel (`app/admin/documents/page.tsx`)

**Authentication Section:**
- Bearer token input (password field) with show/hide toggle (Eye icon)
- Admin password input + **"Regenerate" button** — naya JWT auto-generate karo
- Token expired hone par amber warning banner dikhe
- Token localStorage mein save hota hai

**Indexing Status Section:**
- IDLE / RUNNING / SUCCESS / FAILED badge
- Live stats: Vectors, Chunks, Documents, Vector Store, Embeddings, Started, Finished
- Auto-poll every 2 seconds jab job running ho

**Upload Document Section:**
- **"Auto re-index" toggle** (default ON) — upload ke baad automatic reindex
- File picker (`.md`, `.txt`, `.pdf`, max 10 MB)
- Upload success banner (green, CheckCircle icon)
- Error display

**Documents Table:**
- Raw docs folder ke sab files list
- Name, Path, Size (KB), Modified date

---

## All API Endpoints

| Method | Route | Auth | Kaam |
|--------|-------|------|------|
| `GET` | `/` | None | Root message |
| `GET` | `/health` | None | Health check |
| `POST` | `/auth/admin-token` | None | Admin JWT generate |
| `POST` | `/chat` | Optional | Chat (sync) |
| `POST` | `/chat/stream` | Optional | Chat (SSE streaming) |
| `GET` | `/sessions/{id}` | None | Session history |
| `DELETE` | `/sessions/{id}` | None | Session clear |
| `GET` | `/documents` | Admin JWT | File list |
| `GET` | `/documents/indexing-status` | Admin JWT | Indexing status |
| `POST` | `/documents/upload` | Admin JWT | File upload |
| `POST` | `/documents/reindex` | Admin JWT | Trigger reindex |

> Sab routes `/api/v1/` prefix ke saath bhi available hain.

---

## Environment Variables (`.env`)

```env
# App
APP_NAME="Enterprise HR Policy Assistant"
APP_ENV=local
APP_DEBUG=true
API_V1_PREFIX=/api/v1
FRONTEND_ORIGIN=http://localhost:3000

# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Storage
RAW_DOCS_DIR=./data/raw_docs
PROCESSED_DATA_DIR=./data/processed
VECTOR_STORE_PROVIDER=local_json        # local_json | pinecone
VECTOR_STORE_DIR=./data/processed/vector_store

# Pinecone (prod only)
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=hr-policy-assistant
PINECONE_NAMESPACE=local
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Security
JWT_SECRET_KEY=change-me-in-local-only
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_SECONDS=86400   # 24 hours
ADMIN_PASSWORD=admin123                  # Admin UI ke liye

# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=app-password-here
SMTP_FROM=                              # Empty = SMTP_USER use hoga
SMTP_USE_TLS=true
HR_CONTACT_EMAIL=hr@company.com

# Observability
LOG_LEVEL=INFO
ENABLE_TRACING=false
```

---

## How to Run

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-rag.txt    # Optional: Pinecone, LangGraph
copy .env.example .env                 # Phir apni values daalo
python -m scripts.index_documents      # Documents index karo
uvicorn app.main:app --reload          # http://localhost:8000

# Frontend
cd frontend
npm install
npm run dev                            # http://localhost:3000
```

**Admin token generate karna (terminal se):**
```powershell
cd backend
python -c "from app.security.auth import create_access_token; print(create_access_token({'user_id':'admin','role':'admin'}))"
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Custom JWT (no PyJWT) | Zero extra dependency, HMAC-SHA256 sufficient for MVP |
| Local JSON vector store | Pinecone account ke bina bhi kaam kare |
| Mock embedding provider | OpenAI key ke bina offline development |
| `ExtractiveLLMClient` fallback | API key ke bina bhi basic answers milein |
| In-memory SessionStore | Database setup ke bina conversation history |
| LangGraph optional | `langgraph` install nahi ho toh plain Python pipeline |
| SSE for streaming | WebSocket se simpler, unidirectional data ke liye ideal |
| `h-screen` layout | Header + input fixed rahein, sirf messages scroll karein |

---

## Features Added During Development (Session Notes)

1. **Conversation History** — `app/sessions/` package, `session_id` in chat request/response
2. **Auto Re-index after Upload** — Admin UI mein toggle, upload hote hi reindex trigger
3. **Admin Token Regeneration** — `/auth/admin-token` endpoint + frontend UI (expired token handle)
4. **Streaming Chat** — `POST /chat/stream` SSE + `stream_generate()` in LLM clients
5. **Markdown Rendering** — `react-markdown` + `remark-gfm` + `@tailwindcss/typography`
6. **Suggested Questions** — 4 quick-prompt pills below welcome message
7. **Fixed Header** — `h-screen` + `overflow-hidden` on main element
8. **Thinking Indicator Fix** — "Thinking..." pehle token tak dikhe, phir streaming cursor

---

## Test Coverage

```powershell
cd backend
pytest                    # Sab tests
pytest -v                 # Verbose
pytest app/tests/test_sessions.py    # Sirf session tests
```

Test files:
- `test_config.py`, `test_main.py` — foundation
- `test_loaders.py`, `test_chunking.py` — RAG pipeline
- `test_embeddings.py`, `test_vector_store.py`, `test_retriever.py` — RAG storage
- `test_chat_routes.py`, `test_agent_workflow.py`, `test_agent_tools.py` — agent
- `test_security.py`, `test_document_routes.py` — security + admin
- `test_observability.py`, `test_smtp_client.py`, `test_email_compose.py` — infra
- `test_sessions.py` — conversation history (15 unit + API tests)
- `test_evaluation.py` — quality evaluation

---

*Last updated: 2026-05-20 | Model: Claude Sonnet 4.6*
