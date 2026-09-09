import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import httpx
from fastapi import Request
from sqlmodel import Session

from app.core.config import settings
from app.models import User
from app.services.rag import hybrid_search
from app.services.token_metering import check_token_quota, record_token_usage

logger = logging.getLogger(__name__)


def _format_sse_event(event: str, data: str | dict) -> str:
    """Format payload according to the W3C Server-Sent Events specification."""
    if isinstance(data, (dict, list)):
        payload = json.dumps(data)
    else:
        payload = str(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_rag_tokens(
    session: Session,
    user_id: uuid.UUID,
    query: str,
    request: Request,
    top_k: int = 5,
) -> AsyncGenerator[str]:
    """Asynchronously stream tokens for RAG answers with proactive disconnect detection.

    If the client closes the browser tab or aborts the request, this generator halts execution
    immediately, terminating upstream LLM connections and preventing wasted tokens/compute.
    """
    # 1. Hybrid search retrieval
    matched_chunks = hybrid_search(session, user_id=user_id, query=query, top_k=top_k)

    # 2. Emit sources metadata event
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
    yield _format_sse_event("sources", sources_data)

    if not matched_chunks:
        yield _format_sse_event(
            "token",
            "No relevant information was found in your indexed documents to answer this query.",
        )
        yield _format_sse_event("done", {"status": "completed", "total_tokens": 0})
        return

    # Prepare context
    context_sections = [
        f"[{i}] ({m.document_title}):\n{m.content}"
        for i, m in enumerate(matched_chunks, start=1)
    ]
    context_str = "\n\n".join(context_sections)

    # 3. Live LLM streaming if API key is provided
    prompt_tokens = max(1, len(query) // 4) + sum(
        max(1, len(m.content) // 4) for m in matched_chunks
    )

    if settings.OPENAI_API_KEY:
        system_prompt = (
            "You are an intelligent knowledge assistant. "
            "Answer the user's question accurately based ONLY on the provided context below. "
            "Include inline bracket citations like [1] or [2] matching the provided sources.\n\n"
            f"--- CONTEXT ---\n{context_str}"
        )
        try:
            tokens_streamed = 0
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
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
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        # Critical: Check client disconnect before processing next token
                        if await request.is_disconnected():
                            logger.info(
                                "Client aborted connection. Halting upstream LLM stream."
                            )
                            record_token_usage(
                                session=session,
                                user_id=user_id,
                                model_name="gpt-4o-mini",
                                prompt_tokens=prompt_tokens,
                                completion_tokens=tokens_streamed,
                            )
                            return

                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_content = line[6:].strip()
                        if data_content == "[DONE]":
                            break

                        try:
                            chunk_json = json.loads(data_content)
                            delta = chunk_json["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token:
                                tokens_streamed += 1
                                yield _format_sse_event("token", token)
                        except json.JSONDecodeError, KeyError:
                            continue

            record_token_usage(
                session=session,
                user_id=user_id,
                model_name="gpt-4o-mini",
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_streamed,
            )
            yield _format_sse_event(
                "done", {"status": "completed", "total_tokens": tokens_streamed}
            )
            return
        except Exception as e:
            logger.warning(
                "Upstream streaming failed: %s. Falling back to synthesis.", e
            )

    # 4. Offline / deterministic token stream for tests and local dev
    top_doc_titles = ", ".join({f"'{m.document_title}'" for m in matched_chunks})
    simulated_text = (
        f"Based on your documents ({top_doc_titles}), here is the relevant information: "
        + " ".join(m.content.strip() for m in matched_chunks[:2])
    )
    words = simulated_text.split(" ")
    tokens_sent = 0

    for idx, word in enumerate(words):
        # Disconnect check
        if await request.is_disconnected():
            logger.info("Client aborted connection during stream playback.")
            record_token_usage(
                session=session,
                user_id=user_id,
                model_name="gpt-4o-mini",
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_sent,
            )
            return

        token_to_send = word + (" " if idx < len(words) - 1 else "")
        tokens_sent += 1
        yield _format_sse_event("token", token_to_send)
        # Yield control briefly to event loop for realistic pacing and cancellation checks
        await asyncio.sleep(0.005)

    record_token_usage(
        session=session,
        user_id=user_id,
        model_name="gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens_sent,
    )
    yield _format_sse_event(
        "done", {"status": "completed", "total_tokens": tokens_sent}
    )


async def stream_chat_session_tokens(
    session: Session,
    user: User,
    session_id: uuid.UUID,
    user_query: str,
    request: Request,
    top_k: int = 5,
) -> AsyncGenerator[str]:
    """Asynchronously stream tokens for multi-turn RAG answer within a chat session."""
    # 1. Quota check
    check_token_quota(session, user, estimated_tokens=100)

    # 2. Get chat session & prior messages
    from app.services.chat_memory import (
        add_chat_message,
        format_conversation_history,
        get_chat_session_with_messages,
    )

    chat_session = get_chat_session_with_messages(
        session, session_id=session_id, user_id=user.id, is_superuser=user.is_superuser
    )
    if not chat_session:
        yield _format_sse_event("error", {"detail": "Chat session not found"})
        return

    prior_messages = sorted(
        chat_session.messages or [],
        key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC),
    )
    history_str = format_conversation_history(prior_messages, max_turns=6)

    # 3. Save current user message
    add_chat_message(session, session_id, role="user", content=user_query)

    # 4. Contextual hybrid search query
    search_query = user_query
    if prior_messages:
        last_user_msgs = [m.content for m in prior_messages if m.role == "user"]
        if last_user_msgs and len(user_query.split()) < 5:
            search_query = f"{last_user_msgs[-1]} {user_query}"

    matched_chunks = hybrid_search(
        session, user_id=user.id, query=search_query, top_k=top_k
    )

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
    yield _format_sse_event("sources", sources_data)

    prompt_tokens = (
        max(1, len(user_query) // 4)
        + max(0, len(history_str) // 4)
        + sum(max(1, len(m.content) // 4) for m in matched_chunks)
    )

    if not matched_chunks:
        no_info_msg = "I could not find any relevant information in your documents to answer this question."
        add_chat_message(
            session, session_id, role="assistant", content=no_info_msg, sources=[]
        )
        record_token_usage(
            session,
            user.id,
            "gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=len(no_info_msg) // 4,
        )
        yield _format_sse_event("token", no_info_msg)
        yield _format_sse_event("done", {"status": "completed", "total_tokens": 0})
        return

    context_sections = [
        f"[{i}] (Document: '{m.document_title}', Chunk #{m.chunk_index}):\n{m.content}"
        for i, m in enumerate(matched_chunks, start=1)
    ]
    context_str = "\n\n".join(context_sections)

    # 5. Live LLM streaming if API key is provided
    if settings.OPENAI_API_KEY:
        system_prompt = (
            "You are an intelligent knowledge assistant having a conversation with the user. "
            "Answer the user's question accurately based ONLY on the provided context below. "
            "Maintain conversational continuity using the conversation history. "
            "Include inline bracket citations like [1] or [2] matching the sources.\n\n"
            f"--- CONVERSATION HISTORY ---\n{history_str}\n\n"
            f"--- CONTEXT ---\n{context_str}"
        )
        try:
            tokens_streamed = 0
            collected_text: list[str] = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
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
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if await request.is_disconnected():
                            logger.info("Client disconnected during multi-turn stream.")
                            final_text = "".join(collected_text)
                            if final_text:
                                add_chat_message(
                                    session,
                                    session_id,
                                    role="assistant",
                                    content=final_text,
                                    sources=sources_data,
                                )
                            record_token_usage(
                                session,
                                user.id,
                                "gpt-4o-mini",
                                prompt_tokens=prompt_tokens,
                                completion_tokens=tokens_streamed,
                            )
                            return

                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_content = line[6:].strip()
                        if data_content == "[DONE]":
                            break

                        try:
                            chunk_json = json.loads(data_content)
                            delta = chunk_json["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token:
                                tokens_streamed += 1
                                collected_text.append(token)
                                yield _format_sse_event("token", token)
                        except json.JSONDecodeError, KeyError:
                            continue

            final_text = "".join(collected_text)
            add_chat_message(
                session,
                session_id,
                role="assistant",
                content=final_text,
                sources=sources_data,
            )
            record_token_usage(
                session,
                user.id,
                "gpt-4o-mini",
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_streamed,
            )
            yield _format_sse_event(
                "done", {"status": "completed", "total_tokens": tokens_streamed}
            )
            return
        except Exception as e:
            logger.warning(
                "Upstream multi-turn streaming failed: %s. Falling back to synthesis.",
                e,
            )

    # 6. Offline / deterministic fallback
    top_doc_titles = ", ".join({f"'{m.document_title}'" for m in matched_chunks})
    turn_note = (
        f" (Continuing conversation on turn {len(prior_messages) // 2 + 1})"
        if prior_messages
        else ""
    )
    simulated_text = (
        f"Based on your documents ({top_doc_titles}){turn_note}, here is the relevant answer: "
        + " ".join(m.content.strip() for m in matched_chunks[:2])
    )
    words = simulated_text.split(" ")
    tokens_sent = 0
    collected_words: list[str] = []

    for idx, word in enumerate(words):
        if await request.is_disconnected():
            logger.info("Client aborted connection during multi-turn stream.")
            partial_text = " ".join(collected_words)
            if partial_text:
                add_chat_message(
                    session,
                    session_id,
                    role="assistant",
                    content=partial_text,
                    sources=sources_data,
                )
            record_token_usage(
                session,
                user.id,
                "gpt-4o-mini",
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_sent,
            )
            return

        token_to_send = word + (" " if idx < len(words) - 1 else "")
        tokens_sent += 1
        collected_words.append(word)
        yield _format_sse_event("token", token_to_send)
        await asyncio.sleep(0.005)

    add_chat_message(
        session,
        session_id,
        role="assistant",
        content=simulated_text,
        sources=sources_data,
    )
    record_token_usage(
        session,
        user.id,
        "gpt-4o-mini",
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens_sent,
    )
    yield _format_sse_event(
        "done", {"status": "completed", "total_tokens": tokens_sent}
    )
