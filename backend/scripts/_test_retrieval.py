"""Quick retrieval smoke test — run after indexing to verify vector store."""
from app.rag.embeddings import build_embedding_provider
from app.rag.vector_store import build_vector_store
from app.rag.retriever import RagRetriever
from app.config import get_settings

settings = get_settings()
emb = build_embedding_provider(
    settings.llm_provider,
    openai_api_key=settings.openai_api_key,
    openai_embedding_model=settings.openai_embedding_model,
)
vs = build_vector_store(settings.vector_store_provider, directory=settings.vector_store_dir)
retriever = RagRetriever(embedding_provider=emb, vector_store=vs)

queries = [
    "sick leave policy India",
    "maternity leave how many days",
    "reimbursement laptop expense",
    "work from home WFH guidelines",
]

print(f"Vector store: {settings.vector_store_provider} | {settings.vector_store_dir}")
print("-" * 60)
for query in queries:
    result = retriever.retrieve(query, top_k=1)
    if result.chunks:
        c = result.chunks[0]
        src = c.metadata.get("source", "?")
        print(f"Q: {query}")
        print(f"   source={src}  score={c.score:.3f}")
        print(f"   {c.content[:90]}...")
    else:
        print(f"Q: {query}  --> NO RESULTS")
    print()
