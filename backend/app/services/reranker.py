import logging
import re
from typing import Any

import httpx

from app.core.config import settings
from app.models import RAGChunkMatch

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Clean and tokenize text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def _calculate_cross_relevance_score(
    query: str, content: str, first_stage_score: float
) -> float:
    """Calculate cross-attention relevance score between query and candidate chunk content.

    Evaluates:
    1. Exact phrase alignment and n-gram overlap.
    2. Token overlap and term coverage density.
    3. Positional proximity of query terms within chunk.
    4. First-stage hybrid search confidence.
    """
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)

    if not query_tokens or not content_tokens:
        return first_stage_score

    query_text_lower = query.lower().strip()
    content_text_lower = content.lower()

    # 1. Exact phrase matching bonus
    exact_phrase_bonus = 0.0
    if query_text_lower in content_text_lower:
        exact_phrase_bonus = 0.40
    else:
        # Check 2-word and 3-word n-gram matches
        if len(query_tokens) >= 2:
            ngrams = [
                " ".join(query_tokens[i : i + 2]) for i in range(len(query_tokens) - 1)
            ]
            matched_ngrams = sum(1 for ng in ngrams if ng in content_text_lower)
            exact_phrase_bonus = min(0.30, 0.15 * matched_ngrams)

    # 2. Token overlap & coverage (what % of query words appear in chunk)
    content_token_set = set(content_tokens)
    matched_tokens = [t for t in query_tokens if t in content_token_set]
    coverage_ratio = len(matched_tokens) / len(query_tokens)

    # Term frequency in chunk (bonus for density)
    term_density = sum(content_tokens.count(t) for t in set(matched_tokens)) / len(
        content_tokens
    )
    density_score = min(0.20, term_density * 5.0)

    # 3. Positional proximity (are query words clustered close together?)
    proximity_bonus = 0.0
    if len(matched_tokens) >= 2:
        indices = [i for i, t in enumerate(content_tokens) if t in set(matched_tokens)]
        if indices and len(indices) >= 2:
            span = max(indices) - min(indices) + 1
            if span <= len(query_tokens) * 3:
                proximity_bonus = 0.15

    # 4. Composite cross-encoder score
    cross_score = (
        (0.40 * coverage_ratio) + exact_phrase_bonus + density_score + proximity_bonus
    )

    # Blend 70% cross-score + 30% first-stage hybrid score
    final_score = (0.70 * cross_score) + (0.30 * min(1.0, first_stage_score))
    return round(min(1.0, max(0.0, final_score)), 4)


def rerank_chunks(
    query: str,
    chunks: list[RAGChunkMatch],
    top_k: int = 5,
) -> list[RAGChunkMatch]:
    """Re-rank candidate chunks using cross-attention scoring.

    Supports Cohere Rerank API if COHERE_API_KEY is configured, otherwise
    uses the high-precision deterministic cross-attention scorer.
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        return [
            RAGChunkMatch(
                chunk_id=chunks[0].chunk_id,
                document_id=chunks[0].document_id,
                document_title=chunks[0].document_title,
                chunk_index=chunks[0].chunk_index,
                content=chunks[0].content,
                score=chunks[0].score,
                match_type="reranked",
            )
        ]

    # 1. External Cohere Rerank API if configured
    if settings.COHERE_API_KEY:
        try:
            resp = httpx.post(
                "https://api.cohere.com/v2/rerank",
                headers={
                    "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "rerank-v3.5",
                    "query": query,
                    "documents": [c.content for c in chunks],
                    "top_n": top_k,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            reranked_results: list[RAGChunkMatch] = []
            for item in data.get("results", []):
                idx = item["index"]
                relevance_score = float(item["relevance_score"])
                original_chunk = chunks[idx]
                reranked_results.append(
                    RAGChunkMatch(
                        chunk_id=original_chunk.chunk_id,
                        document_id=original_chunk.document_id,
                        document_title=original_chunk.document_title,
                        chunk_index=original_chunk.chunk_index,
                        content=original_chunk.content,
                        score=round(relevance_score, 4),
                        match_type="reranked",
                    )
                )
            return reranked_results
        except Exception as e:
            logger.warning(
                "Cohere reranking API call failed: %s. Falling back to local cross-encoder.",
                e,
            )

    # 2. Local Cross-Encoder Relevance Scorer
    scored_chunks: list[tuple[float, RAGChunkMatch]] = []
    for chunk in chunks:
        new_score = _calculate_cross_relevance_score(
            query=query, content=chunk.content, first_stage_score=chunk.score
        )
        updated_match = RAGChunkMatch(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=new_score,
            match_type="reranked",
        )
        scored_chunks.append((new_score, updated_match))

    # Sort descending by re-ranked score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]
