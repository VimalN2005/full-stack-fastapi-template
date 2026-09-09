import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Document,
    DocumentChunk,
    DocumentCreate,
    DocumentPublic,
    DocumentsPublic,
    Message,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)
from app.services.rag import generate_rag_answer, hybrid_search, ingest_document

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/documents", response_model=DocumentPublic)
def create_document(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    document_in: DocumentCreate,
) -> Any:
    """Upload and ingest a document into the user's private knowledge base.

    Automatically chunks the document, computes embeddings, and indexes for hybrid search.
    """
    doc = ingest_document(
        session=session,
        user_id=current_user.id,
        title=document_in.title,
        content=document_in.content,
        content_type=document_in.content_type,
    )
    chunk_count = len(doc.chunks) if doc.chunks else 0
    return DocumentPublic(
        id=doc.id,
        title=doc.title,
        content_type=doc.content_type,
        owner_id=doc.owner_id,
        created_at=doc.created_at,
        chunk_count=chunk_count,
    )


@router.get("/documents", response_model=DocumentsPublic)
def read_documents(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Retrieve the current user's indexed documents."""
    count_statement = (
        select(func.count())
        .select_from(Document)
        .where(Document.owner_id == current_user.id)
    )
    count = session.exec(count_statement).one()

    statement = (
        select(Document)
        .where(Document.owner_id == current_user.id)
        .order_by(col(Document.created_at).desc())
        .offset(skip)
        .limit(limit)
    )
    docs = session.exec(statement).all()

    # Query chunk counts for these documents
    doc_ids = [d.id for d in docs]
    counts_map: dict[uuid.UUID, int] = {}
    if doc_ids:
        chunk_counts_stmt = (
            select(DocumentChunk.document_id, func.count(DocumentChunk.id))
            .where(DocumentChunk.document_id.in_(doc_ids))  # type: ignore[attr-defined]
            .group_by(DocumentChunk.document_id)
        )
        for doc_id, c_count in session.exec(chunk_counts_stmt).all():
            counts_map[doc_id] = c_count

    data = [
        DocumentPublic(
            id=d.id,
            title=d.title,
            content_type=d.content_type,
            owner_id=d.owner_id,
            created_at=d.created_at,
            chunk_count=counts_map.get(d.id, 0),
        )
        for d in docs
    ]
    return DocumentsPublic(data=data, count=count)


@router.get("/documents/{id}", response_model=DocumentPublic)
def read_document(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """Get document details by ID (tenant-restricted)."""
    doc = session.get(Document, id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    chunk_count = len(doc.chunks) if doc.chunks else 0
    return DocumentPublic(
        id=doc.id,
        title=doc.title,
        content_type=doc.content_type,
        owner_id=doc.owner_id,
        created_at=doc.created_at,
        chunk_count=chunk_count,
    )


@router.delete("/documents/{id}", response_model=Message)
def delete_document(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """Delete a document and all its associated chunks and vector embeddings."""
    doc = session.get(Document, id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    session.delete(doc)
    session.commit()
    return Message(message="Document and indexed vectors deleted successfully")


@router.post("/search", response_model=RAGSearchResponse)
def search_knowledge_base(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    request: RAGSearchRequest,
) -> Any:
    """Search the user's documents using hybrid search (pgvector cosine similarity + Full-Text Search).

    Returns top-k matching chunks ranked via Reciprocal Rank Fusion (RRF).
    """
    matches = hybrid_search(
        session=session,
        user_id=current_user.id,
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )
    return RAGSearchResponse(
        query=request.query,
        results=matches,
        total=len(matches),
    )


@router.post("/query", response_model=RAGQueryResponse)
def query_knowledge_base(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    request: RAGQueryRequest,
) -> Any:
    """Execute complete RAG pipeline: retrieves relevant chunks and synthesizes a grounded answer with citations."""
    answer, sources = generate_rag_answer(
        session=session,
        user_id=current_user.id,
        query=request.query,
        top_k=request.top_k,
    )
    return RAGQueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
    )
