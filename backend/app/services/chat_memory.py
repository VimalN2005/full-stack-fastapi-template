import json
import logging
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import HTTPException, status
from sqlmodel import Session, col, func, select

from app.core.config import settings
from app.models import (
    ChatMessage,
    ChatSession,
    RAGChunkMatch,
    User,
)
from app.services.rag import hybrid_search
from app.services.token_metering import check_token_quota, record_token_usage

logger = logging.getLogger(__name__)


def create_chat_session(
    session: Session,
    user_id: uuid.UUID,
    title: str | None = None,
) -> ChatSession:
    """Create a new chat session for a user."""
    chat_session = ChatSession(
        user_id=user_id,
        title=title.strip() if title and title.strip() else "New Chat",
    )
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


def get_user_sessions(
    session: Session,
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[ChatSession], int]:
    """Retrieve all chat sessions for a user with total count."""
    count_stmt = (
        select(func.count())
        .select_from(ChatSession)
        .where(ChatSession.user_id == user_id)
    )
    total = session.exec(count_stmt).one()

    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(col(ChatSession.updated_at).desc())
        .offset(skip)
        .limit(limit)
    )
    sessions = session.exec(stmt).all()
    return list(sessions), total


def get_chat_session_with_messages(
    session: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    is_superuser: bool = False,
) -> ChatSession | None:
    """Retrieve a chat session and its full message history with tenant verification."""
    chat_session = session.get(ChatSession, session_id)
    if not chat_session:
        return None
    if chat_session.user_id != user_id and not is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this chat session",
        )
    return chat_session


def delete_chat_session(
    session: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    is_superuser: bool = False,
) -> bool:
    """Delete a chat session and cascade delete all its messages."""
    chat_session = session.get(ChatSession, session_id)
    if not chat_session:
        return False
    if chat_session.user_id != user_id and not is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this chat session",
        )
    session.delete(chat_session)
    session.commit()
    return True


def add_chat_message(
    session: Session,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> ChatMessage:
    """Persist a new message to a chat session and bump the session's updated_at timestamp."""
    sources_json = json.dumps(sources) if sources else None
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources_json,
    )
    session.add(msg)

    # Update session updated_at
    chat_session = session.get(ChatSession, session_id)
    if chat_session:
        chat_session.updated_at = datetime.now(UTC)
        # Update title from first user message if still default
        if chat_session.title == "New Chat" and role == "user":
            snippet = content.strip().split("\n")[0][:45]
            chat_session.title = snippet if snippet else "New Chat"
        session.add(chat_session)

    session.commit()
    session.refresh(msg)
    return msg


def format_conversation_history(
    messages: list[ChatMessage],
    max_turns: int = 6,
) -> str:
    """Format the last N conversation turns for LLM prompt context."""
    # Each turn has 1 user + 1 assistant message = 2 messages per turn
    max_messages = max_turns * 2
    recent_messages = (
        messages[-max_messages:] if len(messages) > max_messages else messages
    )

    history_lines: list[str] = []
    for m in recent_messages:
        speaker = "User" if m.role == "user" else "Assistant"
        history_lines.append(f"{speaker}: {m.content}")

    return "\n".join(history_lines)


def generate_multi_turn_answer(
    session: Session,
    user: User,
    session_id: uuid.UUID,
    user_query: str,
    top_k: int = 5,
) -> tuple[str, list[RAGChunkMatch], ChatMessage, ChatMessage]:
    """Execute multi-turn conversational RAG:

    1. Checks token quota.
    2. Retrieves conversation history.
    3. Saves user message.
    4. Executes pgvector hybrid search using query + history context.
    5. Synthesizes grounded answer.
    6. Saves assistant response with source citations.
    7. Records token usage.
    """
    # 1. Enforce quota guardrail
    check_token_quota(session, user, estimated_tokens=100)

    # 2. Get chat session & prior history
    chat_session = get_chat_session_with_messages(
        session, session_id=session_id, user_id=user.id, is_superuser=user.is_superuser
    )
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    prior_messages = sorted(
        chat_session.messages or [],
        key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC),
    )
    history_str = format_conversation_history(prior_messages, max_turns=6)

    # 3. Save current user message
    user_msg = add_chat_message(session, session_id, role="user", content=user_query)

    # 4. Contextual hybrid search query
    search_query = user_query
    if prior_messages:
        # Include key terms from previous user question if current query is short (e.g. "explain more")
        last_user_msgs = [m.content for m in prior_messages if m.role == "user"]
        if last_user_msgs and len(user_query.split()) < 5:
            search_query = f"{last_user_msgs[-1]} {user_query}"

    matched_chunks = hybrid_search(
        session=session,
        user_id=user.id,
        query=search_query,
        top_k=top_k,
    )

    # 5. Build context sections
    context_sections = [
        f"[{i}] (Document: '{m.document_title}', Chunk #{m.chunk_index}):\n{m.content}"
        for i, m in enumerate(matched_chunks, start=1)
    ]
    context_str = "\n\n".join(context_sections)

    # 6. Synthesize grounded answer
    answer = ""
    if settings.OPENAI_API_KEY:
        try:
            system_prompt = (
                "You are an intelligent knowledge assistant having a conversation with the user. "
                "Answer the user's question accurately based ONLY on the provided context below. "
                "Maintain conversational continuity using the conversation history. "
                "Include inline bracket citations like [1] or [2] matching the sources.\n\n"
                f"--- CONVERSATION HISTORY ---\n{history_str}\n\n"
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
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.2,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(
                "OpenAI synthesis failed: %s. Falling back to local synthesis.", e
            )

    if not answer:
        if matched_chunks:
            top_docs = ", ".join({f"'{m.document_title}'" for m in matched_chunks})
            turn_note = (
                f" (Continuing conversation on turn {len(prior_messages) // 2 + 1})"
                if prior_messages
                else ""
            )
            answer = (
                f"Based on your documents ({top_docs}){turn_note}, here is the relevant answer:\n\n"
                + "\n\n".join(f"• {m.content.strip()}" for m in matched_chunks[:2])
            )
        else:
            answer = "I could not find any relevant information in your documents to answer this question."

    # 7. Format sources for persistence
    sources_data = [
        {
            "chunk_id": str(m.chunk_id),
            "document_id": str(m.document_id),
            "document_title": m.document_title,
            "chunk_index": m.chunk_index,
            "score": m.score,
            "match_type": m.match_type,
        }
        for m in matched_chunks
    ]

    # Save assistant message
    assistant_msg = add_chat_message(
        session, session_id, role="assistant", content=answer, sources=sources_data
    )

    # 8. Record token usage
    prompt_tokens = (
        max(1, len(user_query) // 4)
        + max(0, len(history_str) // 4)
        + sum(max(1, len(m.content) // 4) for m in matched_chunks)
    )
    completion_tokens = max(1, len(answer) // 4)
    record_token_usage(
        session=session,
        user_id=user.id,
        model_name="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return answer, matched_chunks, user_msg, assistant_msg
