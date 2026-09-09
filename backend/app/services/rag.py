import math
import re
import uuid

import httpx
from sqlalchemy import func
from sqlmodel import Session, col, select

from app.core.config import settings
from app.models import Document, DocumentChunk, RAGChunkMatch
from app.services.embeddings import embedding_service


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into manageable chunks using sentence and paragraph boundaries with overlap."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Split into paragraphs or sentences
    paragraphs = re.split(r"\n\s*\n", cleaned)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)
        if current_len + para_len + 1 > chunk_size and current_chunk:
            combined = "\n\n".join(current_chunk)
            chunks.append(combined)

            # Preserve overlap from end of current chunk
            if chunk_overlap > 0 and len(combined) > chunk_overlap:
                overlap_text = combined[-chunk_overlap:]
                current_chunk = [overlap_text, para]
                current_len = len(overlap_text) + para_len + 1
            else:
                current_chunk = [para]
                current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 1

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def ingest_document(
    session: Session,
    user_id: uuid.UUID,
    title: str,
    content: str,
    content_type: str = "text/plain",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Document:
    """Ingest a new document, create chunks, compute embeddings, and persist with tenant isolation."""
    # 1. Create and persist the parent document
    doc = Document(
        title=title,
        content_type=content_type,
        owner_id=user_id,
    )
    session.add(doc)
    session.flush()  # Populates doc.id

    # 2. Split content into chunks
    text_chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
    if not text_chunks:
        text_chunks = [content.strip() or "Empty document"]

    # 3. Compute vector embeddings in batch
    embeddings = embedding_service.get_embeddings(text_chunks)

    # 4. Save chunks with strict tenant-isolation foreign key (owner_id)
    chunks_to_create = []
    for idx, (chunk_text, vector) in enumerate(
        zip(text_chunks, embeddings, strict=False)
    ):
        chunk = DocumentChunk(
            document_id=doc.id,
            owner_id=user_id,  # Denormalized for ultra-fast indexed multi-tenant filtering
            chunk_index=idx,
            content=chunk_text,
            embedding=vector,
        )
        chunks_to_create.append(chunk)

    session.add_all(chunks_to_create)
    session.commit()
    session.refresh(doc)
    return doc


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors in Python."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def hybrid_search(
    session: Session,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    rrf_k: int = 60,
) -> list[RAGChunkMatch]:
    """Execute hybrid search (Dense Vector + Full-Text Search) with Reciprocal Rank Fusion (RRF).

    Enforces strict tenant isolation: only chunks where chunk.owner_id == user_id are searched.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    query_vector = embedding_service.get_embedding(cleaned_query)

    # Check if database backend supports native pgvector cosine distance
    is_postgres = (
        session.bind is not None and "postgres" in session.bind.dialect.name.lower()
    )

    dense_ranked: list[tuple[DocumentChunk, float]] = []
    keyword_ranked: list[tuple[DocumentChunk, float]] = []

    if is_postgres:
        try:
            # 1. Dense Semantic Search via pgvector (<=> cosine distance operator)
            # Distance: 0 = identical, 2 = opposite. Similarity = 1 - distance
            cosine_dist = DocumentChunk.embedding.cosine_distance(query_vector)  # type: ignore[attr-defined]
            dense_stmt = (
                select(DocumentChunk, cosine_dist.label("distance"))
                .where(DocumentChunk.owner_id == user_id)
                .where(DocumentChunk.embedding.is_not(None))
                .order_by("distance")
                .limit(top_k * 2)
            )
            dense_rows = session.exec(dense_stmt).all()
            for chunk, dist in dense_rows:
                sim = max(0.0, min(1.0, 1.0 - float(dist or 0.0)))
                dense_ranked.append((chunk, sim))

            # 2. Full-Text Search via PostgreSQL tsvector & plainto_tsquery
            fts_query = func.plainto_tsquery("english", cleaned_query)
            fts_vector = func.to_tsvector("english", DocumentChunk.content)
            rank = func.ts_rank_cd(fts_vector, fts_query)
            fts_stmt = (
                select(DocumentChunk, rank.label("score"))
                .where(DocumentChunk.owner_id == user_id)
                .where(fts_vector.op("@@")(fts_query))
                .order_by(rank.desc())
                .limit(top_k * 2)
            )
            fts_rows = session.exec(fts_stmt).all()
            for chunk, score in fts_rows:
                keyword_ranked.append((chunk, float(score or 0.0)))
        except Exception:
            # Fallback to python evaluation if SQL functions encounter syntax issues
            dense_ranked = []
            keyword_ranked = []

    # Fallback to in-memory evaluation (e.g. SQLite or when offline/testing)
    if not dense_ranked and not keyword_ranked:
        all_user_chunks = session.exec(
            select(DocumentChunk).where(DocumentChunk.owner_id == user_id)
        ).all()

        query_terms = set(re.findall(r"\w+", cleaned_query.lower()))

        temp_dense = []
        temp_keyword = []
        for c in all_user_chunks:
            # Vector similarity
            if c.embedding:
                sim = (_cosine_similarity(query_vector, c.embedding) + 1.0) / 2.0
                temp_dense.append((c, sim))

            # Simple token match keyword score
            c_terms = set(re.findall(r"\w+", c.content.lower()))
            overlap = len(query_terms.intersection(c_terms))
            if overlap > 0:
                score = overlap / max(1, len(query_terms))
                temp_keyword.append((c, score))

        temp_dense.sort(key=lambda x: x[1], reverse=True)
        temp_keyword.sort(key=lambda x: x[1], reverse=True)
        dense_ranked = temp_dense[: top_k * 2]
        keyword_ranked = temp_keyword[: top_k * 2]

    # Reciprocal Rank Fusion (RRF)
    # RRF(d) = sum(1 / (k + rank_i(d)))
    rrf_scores: dict[uuid.UUID, float] = {}
    chunk_map: dict[uuid.UUID, DocumentChunk] = {}
    match_types: dict[uuid.UUID, str] = {}

    for rank, (chunk, _score) in enumerate(dense_ranked, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_map[chunk.id] = chunk
        match_types[chunk.id] = "dense"

    for rank, (chunk, _score) in enumerate(keyword_ranked, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_map[chunk.id] = chunk
        if chunk.id in match_types and match_types[chunk.id] == "dense":
            match_types[chunk.id] = "hybrid"
        else:
            match_types[chunk.id] = "keyword"

    # Sort merged results by RRF score
    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )

    # Fetch document titles in batch
    doc_ids = {chunk_map[cid].document_id for cid in sorted_ids}
    doc_titles: dict[uuid.UUID, str] = {}
    if doc_ids:
        docs = session.exec(select(Document).where(col(Document.id).in_(doc_ids))).all()
        doc_titles = {d.id: d.title for d in docs}

    results: list[RAGChunkMatch] = []
    # Normalize top score to 1.0 for user readability
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

    for cid in sorted_ids:
        raw_score = rrf_scores[cid]
        normalized_score = round(raw_score / max_rrf, 4)
        if normalized_score < min_score:
            continue

        chunk = chunk_map[cid]
        results.append(
            RAGChunkMatch(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=doc_titles.get(chunk.document_id, "Untitled Document"),
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=normalized_score,
                match_type=match_types.get(cid, "hybrid"),
            )
        )
        if len(results) >= top_k:
            break

    return results


def generate_rag_answer(
    session: Session,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 5,
) -> tuple[str, list[RAGChunkMatch]]:
    """Execute hybrid search, assemble grounding context, and generate answer with source citations."""
    matched_chunks = hybrid_search(session, user_id=user_id, query=query, top_k=top_k)

    if not matched_chunks:
        return (
            "No relevant information was found in your indexed documents to answer this query.",
            [],
        )

    # Build context string with numbered citations
    context_sections: list[str] = []
    for idx, match in enumerate(matched_chunks, start=1):
        context_sections.append(
            f"[{idx}] (Document: '{match.document_title}', Chunk #{match.chunk_index}):\n{match.content}"
        )
    context_str = "\n\n".join(context_sections)

    # If OpenAI API Key is configured, use LLM completion
    if settings.OPENAI_API_KEY:
        try:
            system_prompt = (
                "You are an intelligent knowledge assistant. "
                "Answer the user's question accurately based ONLY on the provided context below. "
                "Include inline bracket citations like [1] or [2] matching the provided sources. "
                "If the context does not contain sufficient details to answer, state that clearly.\n\n"
                f"--- CONTEXT ---\n{context_str}"
            )
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.2,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return answer, matched_chunks
        except Exception:
            pass

    # Offline / deterministic synthesis fallback
    top_sources = ", ".join({f"'{m.document_title}'" for m in matched_chunks})
    answer = (
        f"Based on your documents ({top_sources}), here is the relevant synthesized information:\n\n"
        + "\n\n".join(f"• {m.content.strip()}" for m in matched_chunks[:3])
    )
    return answer, matched_chunks
