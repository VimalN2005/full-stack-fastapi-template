import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    ChatMessageCreate,
    ChatMessagePublic,
    ChatSessionCreate,
    ChatSessionDetailPublic,
    ChatSessionPublic,
    ChatSessionsPublic,
    Message,
)
from app.services.chat_memory import (
    create_chat_session,
    delete_chat_session,
    generate_multi_turn_answer,
    get_chat_session_with_messages,
    get_user_sessions,
)
from app.services.streaming import stream_chat_session_tokens

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionPublic)
def create_session(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    session_in: ChatSessionCreate | None = None,
) -> Any:
    """Create a new conversational chat session."""
    title = session_in.title if session_in else None
    chat_session = create_chat_session(
        session=session, user_id=current_user.id, title=title
    )
    return ChatSessionPublic(
        id=chat_session.id,
        user_id=chat_session.user_id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        message_count=0,
    )


@router.get("/sessions", response_model=ChatSessionsPublic)
def list_sessions(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 50,
) -> Any:
    """List all conversational chat sessions for the current user."""
    sessions, total = get_user_sessions(
        session=session, user_id=current_user.id, skip=skip, limit=limit
    )
    data = [
        ChatSessionPublic(
            id=s.id,
            user_id=s.user_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages) if s.messages else 0,
        )
        for s in sessions
    ]
    return ChatSessionsPublic(data=data, count=total)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailPublic)
def get_session(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    session_id: uuid.UUID,
) -> Any:
    """Get chat session details and complete chronological message history."""
    chat_session = get_chat_session_with_messages(
        session=session,
        session_id=session_id,
        user_id=current_user.id,
        is_superuser=current_user.is_superuser,
    )
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages_data = [
        ChatMessagePublic(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            sources=m.sources,
            created_at=m.created_at,
        )
        for m in sorted(chat_session.messages or [], key=lambda x: x.created_at or x.id)
    ]
    return ChatSessionDetailPublic(
        id=chat_session.id,
        user_id=chat_session.user_id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        messages=messages_data,
    )


@router.delete("/sessions/{session_id}", response_model=Message)
def remove_session(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    session_id: uuid.UUID,
) -> Any:
    """Delete a chat session and all its messages."""
    success = delete_chat_session(
        session=session,
        session_id=session_id,
        user_id=current_user.id,
        is_superuser=current_user.is_superuser,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return Message(message="Chat session and messages deleted successfully")


@router.post("/sessions/{session_id}/messages", response_model=ChatMessagePublic)
def send_chat_message(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    session_id: uuid.UUID,
    message_in: ChatMessageCreate,
) -> Any:
    """Send a user message in a chat session and receive grounded, multi-turn RAG answer."""
    answer, sources, user_msg, assistant_msg = generate_multi_turn_answer(
        session=session,
        user=current_user,
        session_id=session_id,
        user_query=message_in.content,
        top_k=message_in.top_k,
    )
    return ChatMessagePublic(
        id=assistant_msg.id,
        session_id=assistant_msg.session_id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        sources=assistant_msg.sources,
        created_at=assistant_msg.created_at,
    )


@router.post("/sessions/{session_id}/stream")
async def stream_chat_message(
    *,
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    session_id: uuid.UUID,
    message_in: ChatMessageCreate,
) -> StreamingResponse:
    """Stream token-by-token multi-turn conversational RAG answer via Server-Sent Events (SSE)."""
    event_stream = stream_chat_session_tokens(
        session=session,
        user=current_user,
        session_id=session_id,
        user_query=message_in.content,
        request=request,
        top_k=message_in.top_k,
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
