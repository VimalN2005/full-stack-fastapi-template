import uuid

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models import User


def test_get_ai_usage_normal_user(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    r = client.get(
        f"{settings.API_V1_STR}/ai/usage",
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert "total_tokens_month" in data
    assert "monthly_limit" in data
    assert "remaining_tokens" in data
    assert "estimated_cost_usd" in data
    assert "usage_percentage" in data
    assert data["is_unlimited"] is False
    assert data["monthly_limit"] >= 0


def test_get_ai_usage_superuser(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    r = client.get(
        f"{settings.API_V1_STR}/ai/usage",
        headers=superuser_token_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_unlimited"] is True
    assert data["usage_percentage"] == 0.0


def test_get_token_history(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    r = client.get(
        f"{settings.API_V1_STR}/ai/history",
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert "data" in data
    assert "count" in data


def test_update_user_quota_superuser(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    # Find normal user
    normal_user = db.exec(
        select(User).where(User.email == settings.EMAIL_TEST_USER)
    ).first()
    assert normal_user is not None

    new_limit = 150000
    r = client.patch(
        f"{settings.API_V1_STR}/ai/users/{normal_user.id}/quota",
        headers=superuser_token_headers,
        json={"monthly_token_limit": new_limit},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["monthly_token_limit"] == new_limit


def test_update_user_quota_forbidden_for_normal_user(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    dummy_id = uuid.uuid4()
    r = client.patch(
        f"{settings.API_V1_STR}/ai/users/{dummy_id}/quota",
        headers=normal_user_token_headers,
        json={"monthly_token_limit": 50000},
    )
    assert r.status_code == 403


def test_rag_query_records_tokens_and_enforces_quota(
    client: TestClient, normal_user_token_headers: dict[str, str], db: Session
) -> None:
    user = db.exec(select(User).where(User.email == settings.EMAIL_TEST_USER)).first()
    assert user is not None

    # 1. Ingest a document
    doc_data = {
        "title": "Quantum Physics Notes",
        "content": "Quantum entanglement occurs when two particles remain connected regardless of distance.",
        "content_type": "text/plain",
    }
    r_ingest = client.post(
        f"{settings.API_V1_STR}/rag/documents",
        headers=normal_user_token_headers,
        json=doc_data,
    )
    assert r_ingest.status_code == 200

    # 2. Query knowledge base
    r_query = client.post(
        f"{settings.API_V1_STR}/rag/query",
        headers=normal_user_token_headers,
        json={"query": "What is quantum entanglement?"},
    )
    assert r_query.status_code == 200

    # 3. Check usage stats updated
    r_usage = client.get(
        f"{settings.API_V1_STR}/ai/usage",
        headers=normal_user_token_headers,
    )
    assert r_usage.status_code == 200
    usage_data = r_usage.json()
    assert usage_data["total_tokens_month"] > 0
    assert usage_data["estimated_cost_usd"] >= 0.0

    # 4. Now exhaust quota by setting limit to 0
    user.monthly_token_limit = 0
    db.add(user)
    db.commit()

    # Query should now be blocked with HTTP 429
    r_blocked = client.post(
        f"{settings.API_V1_STR}/rag/query",
        headers=normal_user_token_headers,
        json={"query": "Will this fail with 429?"},
    )
    assert r_blocked.status_code == 429
    assert "Monthly AI token quota exceeded" in r_blocked.json()["detail"]

    # Restore limit
    user.monthly_token_limit = 50000
    db.add(user)
    db.commit()
