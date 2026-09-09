from fastapi.testclient import TestClient

from app.core.config import settings


def test_async_document_ingestion_and_status(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    # 1. Post document with background=True
    doc_payload = {
        "title": "Large Distributed Systems Whitepaper",
        "content": (
            "Distributed consensus algorithms like Raft and Paxos ensure consistency across replicas. "
            "Leader election handles node failures gracefully with randomized heartbeats."
        ),
        "content_type": "text/plain",
    }
    r_create = client.post(
        f"{settings.API_V1_STR}/rag/documents?background=true",
        headers=superuser_token_headers,
        json=doc_payload,
    )
    # Background ingestion returns 202 Accepted
    assert r_create.status_code == 202
    created_data = r_create.json()
    assert created_data["title"] == doc_payload["title"]
    doc_id = created_data["id"]

    # In testclient, background tasks run automatically before request completes
    # 2. Poll status endpoint
    r_status = client.get(
        f"{settings.API_V1_STR}/rag/documents/{doc_id}/status",
        headers=superuser_token_headers,
    )
    assert r_status.status_code == 200
    status_data = r_status.json()
    assert status_data["id"] == doc_id
    assert status_data["status"] == "ready"
    assert status_data["chunk_count"] >= 1
    assert status_data["error_message"] is None

    # 3. Read specific document
    r_read = client.get(
        f"{settings.API_V1_STR}/rag/documents/{doc_id}",
        headers=superuser_token_headers,
    )
    assert r_read.status_code == 200
    read_data = r_read.json()
    assert read_data["status"] == "ready"
    assert read_data["chunk_count"] >= 1

    # Cleanup
    client.delete(
        f"{settings.API_V1_STR}/rag/documents/{doc_id}",
        headers=superuser_token_headers,
    )
