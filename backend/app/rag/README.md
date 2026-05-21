# RAG Module

Document ingestion, chunking, embeddings, vector storage, retrieval, and reranking will live here.

Planned files:

- `loaders.py`
- `chunking.py`
- `embeddings.py`
- `vector_store.py`
- `retriever.py`
- `reranker.py`

Current module:

- `loaders.py` loads local Markdown, text, and PDF documents.
- Markdown front matter can provide metadata such as `country`, `employee_type`, `policy_type`, and `access_level`.
- Loader output is normalized into `LoadedDocument` with `DocumentMetadata`.
- `chunking.py` converts loaded documents into section-aware `DocumentChunk` objects.
- Chunk metadata preserves document fields and adds chunk index, section title, and character offsets.
- `embeddings.py` provides OpenAI and deterministic mock embedding providers.
- `vector_store.py` provides local JSON storage for tests/MVP and an optional Pinecone wrapper.
- `retriever.py` embeds user queries, runs top-k vector search, applies metadata filters, filters low scores, and formats context for prompts.
