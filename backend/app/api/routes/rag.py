import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Document,
    DocumentChunk,
    DocumentCreate,
    DocumentPublic,
    DocumentsPublic,
    DocumentStatusResponse,
    Message,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)
from app.services.rag import (
    generate_rag_answer,
    hybrid_search,
    ingest_document,
    process_document_in_background,
)
from app.services.streaming import stream_rag_tokens
from app.services.token_metering import check_token_quota, record_token_usage

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/documents", response_model=DocumentPublic)
def create_document(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    document_in: DocumentCreate,
    background_tasks: BackgroundTasks,
    response: Response,
    background: bool = False,
) -> Any:
    """Upload and ingest a document into the user's private knowledge base.

    If background=True, creates a processing placeholder, enqueues ingestion to background tasks,
    and returns HTTP 202 Accepted.
    """
    embed_tokens = max(1, len(document_in.content) // 4)
    check_token_quota(session, current_user, estimated_tokens=embed_tokens)

    if background:
        doc = Document(
            title=document_in.title,
            content_type=document_in.content_type,
            status="processing",
            owner_id=current_user.id,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)

        background_tasks.add_task(
            process_document_in_background,
            doc_id=doc.id,
            user_id=current_user.id,
            content=document_in.content,
            engine=session.get_bind(),
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return DocumentPublic(
            id=doc.id,
            title=doc.title,
            content_type=doc.content_type,
            status=doc.status,
            owner_id=doc.owner_id,
            created_at=doc.created_at,
            chunk_count=0,
            error_message=None,
        )

    doc = ingest_document(
        session=session,
        user_id=current_user.id,
        title=document_in.title,
        content=document_in.content,
        content_type=document_in.content_type,
    )
    record_token_usage(
        session=session,
        user_id=current_user.id,
        model_name="text-embedding-3-small",
        prompt_tokens=embed_tokens,
        completion_tokens=0,
    )
    chunk_count = len(doc.chunks) if doc.chunks else 0
    return DocumentPublic(
        id=doc.id,
        title=doc.title,
        content_type=doc.content_type,
        status=doc.status,
        owner_id=doc.owner_id,
        created_at=doc.created_at,
        chunk_count=chunk_count,
        error_message=doc.error_message,
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
            status=d.status,
            owner_id=d.owner_id,
            created_at=d.created_at,
            chunk_count=counts_map.get(d.id, 0),
            error_message=d.error_message,
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
        status=doc.status,
        owner_id=doc.owner_id,
        created_at=doc.created_at,
        chunk_count=chunk_count,
        error_message=doc.error_message,
    )


@router.get("/documents/{id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """Check asynchronous ingestion status and chunk count for a document."""
    doc = session.get(Document, id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    chunk_count = len(doc.chunks) if doc.chunks else 0
    return DocumentStatusResponse(
        id=doc.id,
        title=doc.title,
        status=doc.status,
        chunk_count=chunk_count,
        error_message=doc.error_message,
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
        rerank=request.rerank,
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
    check_token_quota(session, current_user, estimated_tokens=50)

    answer, sources = generate_rag_answer(
        session=session,
        user_id=current_user.id,
        query=request.query,
        top_k=request.top_k,
        rerank=request.rerank,
    )

    prompt_tokens = max(1, len(request.query) // 4) + sum(
        max(1, len(s.content) // 4) for s in sources
    )
    completion_tokens = max(1, len(answer) // 4)
    record_token_usage(
        session=session,
        user_id=current_user.id,
        model_name="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return RAGQueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
    )


@router.post("/stream")
async def stream_knowledge_base(
    *,
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    query_in: RAGQueryRequest,
) -> StreamingResponse:
    """Stream token-by-token RAG answer via Server-Sent Events (SSE).

    Proactively detects client disconnections (e.g. user clicks Stop Generating or closes tab)
    and cancels upstream execution immediately to avoid wasting tokens or compute.
    """
    check_token_quota(session, current_user, estimated_tokens=50)

    event_stream = stream_rag_tokens(
        session=session,
        user_id=current_user.id,
        query=query_in.query,
        request=request,
        top_k=query_in.top_k,
        rerank=query_in.rerank,
    )
    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
