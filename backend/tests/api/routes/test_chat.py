from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models import User


def test_chat_session_crud(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    # 1. Create a session
    r = client.post(
        f"{settings.API_V1_STR}/chat/sessions",
        headers=normal_user_token_headers,
        json={"title": "RAG Research Session"},
    )
    assert r.status_code == 200
    session_data = r.json()
    assert session_data["title"] == "RAG Research Session"
    session_id = session_data["id"]

    # 2. List sessions
    r_list = client.get(
        f"{settings.API_V1_STR}/chat/sessions",
        headers=normal_user_token_headers,
    )
    assert r_list.status_code == 200
    list_data = r_list.json()
    assert list_data["count"] >= 1
    assert any(s["id"] == session_id for s in list_data["data"])

    # 3. Get session detail
    r_get = client.get(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}",
        headers=normal_user_token_headers,
    )
    assert r_get.status_code == 200
    detail_data = r_get.json()
    assert detail_data["id"] == session_id
    assert detail_data["messages"] == []

    # 4. Delete session
    r_del = client.delete(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}",
        headers=normal_user_token_headers,
    )
    assert r_del.status_code == 200

    # Verify 404 after delete
    r_verify = client.get(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}",
        headers=normal_user_token_headers,
    )
    assert r_verify.status_code == 404


def test_multi_turn_conversation_and_streaming(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    # 1. Ingest document for RAG context
    doc_payload = {
        "title": "FastAPI Async Architecture",
        "content": "FastAPI uses Starlette for the web parts and Pydantic for data validation. AnyIO handles concurrency.",
        "content_type": "text/plain",
    }
    r_doc = client.post(
        f"{settings.API_V1_STR}/rag/documents",
        headers=normal_user_token_headers,
        json=doc_payload,
    )
    assert r_doc.status_code == 200

    # 2. Create chat session
    r_sess = client.post(
        f"{settings.API_V1_STR}/chat/sessions",
        headers=normal_user_token_headers,
        json={"title": "FastAPI Async Discussion"},
    )
    assert r_sess.status_code == 200
    session_id = r_sess.json()["id"]

    # 3. Send Turn 1 message (Sync endpoint)
    r_turn1 = client.post(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}/messages",
        headers=normal_user_token_headers,
        json={"content": "What library handles concurrency in FastAPI?", "top_k": 3},
    )
    assert r_turn1.status_code == 200
    turn1_data = r_turn1.json()
    assert turn1_data["role"] == "assistant"
    assert len(turn1_data["content"]) > 0

    # 4. Send Turn 2 message (Streaming endpoint via SSE)
    r_stream = client.post(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}/stream",
        headers=normal_user_token_headers,
        json={"content": "What about the web parts?", "top_k": 3},
    )
    assert r_stream.status_code == 200
    assert "text/event-stream" in r_stream.headers["content-type"]
    assert "event: sources" in r_stream.text
    assert "event: token" in r_stream.text
    assert "event: done" in r_stream.text

    # 5. Check session history now contains 4 messages: Turn 1 (user+assistant), Turn 2 (user+assistant)
    r_history = client.get(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}",
        headers=normal_user_token_headers,
    )
    assert r_history.status_code == 200
    history_data = r_history.json()
    assert len(history_data["messages"]) == 4
    assert [m["role"] for m in history_data["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_chat_session_tenant_isolation(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    normal_user_token_headers: dict[str, str],
) -> None:
    # Superuser creates a session
    r_admin_sess = client.post(
        f"{settings.API_V1_STR}/chat/sessions",
        headers=superuser_token_headers,
        json={"title": "Admin Confidential Session"},
    )
    assert r_admin_sess.status_code == 200
    admin_session_id = r_admin_sess.json()["id"]

    # Normal user tries to access admin session -> 403 Forbidden
    r_forbidden = client.get(
        f"{settings.API_V1_STR}/chat/sessions/{admin_session_id}",
        headers=normal_user_token_headers,
    )
    assert r_forbidden.status_code == 403


def test_chat_message_quota_blocking(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
) -> None:
    user = db.exec(select(User).where(User.email == settings.EMAIL_TEST_USER)).first()
    assert user is not None

    # Create session
    r_sess = client.post(
        f"{settings.API_V1_STR}/chat/sessions",
        headers=normal_user_token_headers,
        json={"title": "Quota Test Session"},
    )
    session_id = r_sess.json()["id"]

    # Exhaust quota
    user.monthly_token_limit = 0
    db.add(user)
    db.commit()

    # Posting message should now fail with 429
    r_blocked = client.post(
        f"{settings.API_V1_STR}/chat/sessions/{session_id}/messages",
        headers=normal_user_token_headers,
        json={"content": "Can I ask something with zero tokens?"},
    )
    assert r_blocked.status_code == 429

    # Restore limit
    user.monthly_token_limit = 50000
    db.add(user)
    db.commit()
