import math
import uuid

from sqlmodel import Session

from app.services.embeddings import EmbeddingService
from app.services.rag import (
    generate_rag_answer,
    hybrid_search,
    ingest_document,
    split_text_into_chunks,
)


def test_split_text_into_chunks() -> None:
    text = (
        "FastAPI is a modern, fast web framework for building APIs with Python.\n\n"
        "It is based on standard Python type hints and Starlette.\n\n"
        "SQLModel is a library for interacting with SQL databases from Python code.\n\n"
        "pgvector adds vector similarity search capabilities directly to PostgreSQL."
    )
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_embedding_service_deterministic() -> None:
    svc = EmbeddingService(dimension=1536)
    v1 = svc.get_embedding("Machine learning with FastAPI and pgvector")
    v2 = svc.get_embedding("Machine learning with FastAPI and pgvector")
    v3 = svc.get_embedding("Completely unrelated culinary recipe for pasta")

    assert len(v1) == 1536
    # Exact match for identical input
    assert v1 == v2

    # L2 Norm should be approximately 1.0
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-4

    # Different text should produce different vectors
    assert v1 != v3


def test_multi_tenant_rag_isolation(db: Session) -> None:
    """Critical Test: Verify that User A cannot retrieve or see User B's documents/chunks."""
    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()

    # User A ingests confidential financial report
    doc_a = ingest_document(
        session=db,
        user_id=user_a_id,
        title="Q3 Confidential Financials",
        content="Our secret net profit for Q3 was 42 million dollars.",
    )

    # User B ingests engineering documentation
    doc_b = ingest_document(
        session=db,
        user_id=user_b_id,
        title="Engineering System Architecture",
        content="Our microservices communicate via gRPC and Redis queues.",
    )

    # 1. User A searches for their secret profit
    results_a = hybrid_search(
        session=db,
        user_id=user_a_id,
        query="net profit secret dollars",
        top_k=5,
    )
    assert len(results_a) > 0
    assert any(r.document_id == doc_a.id for r in results_a)
    # Ensure User A NEVER sees User B's document
    assert not any(r.document_id == doc_b.id for r in results_a)

    # 2. User B queries for User A's secret profit - MUST return zero results
    results_b = hybrid_search(
        session=db,
        user_id=user_b_id,
        query="net profit secret dollars",
        top_k=5,
    )
    assert not any(r.document_id == doc_a.id for r in results_b)

    # 3. User B queries their own engineering docs
    results_b_eng = hybrid_search(
        session=db,
        user_id=user_b_id,
        query="gRPC microservices architecture",
        top_k=5,
    )
    assert len(results_b_eng) > 0
    assert any(r.document_id == doc_b.id for r in results_b_eng)
    assert not any(r.document_id == doc_a.id for r in results_b_eng)

    # 4. Test RAG Q&A synthesis
    answer, sources = generate_rag_answer(
        session=db,
        user_id=user_b_id,
        query="What do microservices use?",
        top_k=3,
    )
    assert len(sources) > 0
    assert "Engineering System Architecture" in answer or "gRPC" in answer

    # Cleanup test records
    db.delete(doc_a)
    db.delete(doc_b)
    db.commit()
