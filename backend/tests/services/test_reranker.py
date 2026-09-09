import uuid

from app.models import RAGChunkMatch
from app.services.reranker import _calculate_cross_relevance_score, rerank_chunks


def test_calculate_cross_relevance_score_bounds() -> None:
    score = _calculate_cross_relevance_score(
        query="FastAPI async endpoints",
        content="FastAPI supports async def endpoints natively.",
        first_stage_score=0.6,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_rerank_promotes_exact_relevance() -> None:
    doc_id = uuid.uuid4()
    query = "How does backpropagation calculate gradients?"

    # Chunk A was ranked #1 by vector search due to general ML terms, but doesn't answer the question
    chunk_a = RAGChunkMatch(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_title="ML Overview",
        chunk_index=0,
        content="Machine learning neural networks have multiple dense layers and activation functions.",
        score=0.85,
        match_type="hybrid",
    )

    # Chunk B was ranked #2 by vector search, but exactly explains backpropagation and gradients
    chunk_b = RAGChunkMatch(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_title="Backprop Deep Dive",
        chunk_index=1,
        content="Backpropagation calculates gradients through the computational graph using the calculus chain rule.",
        score=0.60,
        match_type="hybrid",
    )

    # Chunk C is unrelated
    chunk_c = RAGChunkMatch(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_title="Dataset Prep",
        chunk_index=2,
        content="Data loaders handle image resizing, cropping, and shuffling.",
        score=0.20,
        match_type="hybrid",
    )

    reranked = rerank_chunks(query=query, chunks=[chunk_a, chunk_b, chunk_c], top_k=2)

    assert len(reranked) == 2
    # Chunk B must be promoted to rank 0 because of high cross-attention relevance!
    assert reranked[0].chunk_id == chunk_b.chunk_id
    assert reranked[0].match_type == "reranked"
    assert reranked[0].score > reranked[1].score


def test_rerank_handles_edge_cases() -> None:
    # Empty chunks
    assert rerank_chunks(query="test", chunks=[], top_k=5) == []

    # Single chunk
    single = RAGChunkMatch(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Title",
        chunk_index=0,
        content="Single test chunk",
        score=0.5,
        match_type="hybrid",
    )
    result = rerank_chunks(query="test", chunks=[single], top_k=5)
    assert len(result) == 1
    assert result[0].match_type == "reranked"


def test_rerank_respects_top_k() -> None:
    doc_id = uuid.uuid4()
    chunks = [
        RAGChunkMatch(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            document_title="Doc",
            chunk_index=i,
            content=f"Content piece number {i} mentioning python and fastapi.",
            score=0.5 - (i * 0.05),
            match_type="hybrid",
        )
        for i in range(6)
    ]

    result = rerank_chunks(query="python and fastapi", chunks=chunks, top_k=3)
    assert len(result) == 3
