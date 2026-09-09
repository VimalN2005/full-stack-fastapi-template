import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator

import httpx
from fastapi import Request
from sqlmodel import Session

from app.core.config import settings
from app.services.rag import hybrid_search

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
    if settings.OPENAI_API_KEY:
        system_prompt = (
            "You are an intelligent knowledge assistant. "
            "Answer the user's question accurately based ONLY on the provided context below. "
            "Include inline bracket citations like [1] or [2] matching the provided sources.\n\n"
            f"--- CONTEXT ---\n{context_str}"
        )
        try:
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
                                yield _format_sse_event("token", token)
                        except json.JSONDecodeError, KeyError:
                            continue

            yield _format_sse_event("done", {"status": "completed"})
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

    for idx, word in enumerate(words):
        # Disconnect check
        if await request.is_disconnected():
            logger.info("Client aborted connection during stream playback.")
            return

        token_to_send = word + (" " if idx < len(words) - 1 else "")
        yield _format_sse_event("token", token_to_send)
        # Yield control briefly to event loop for realistic pacing and cancellation checks
        await asyncio.sleep(0.005)

    yield _format_sse_event("done", {"status": "completed", "total_tokens": len(words)})
