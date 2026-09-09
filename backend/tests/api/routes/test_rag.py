import uuid

from fastapi.testclient import TestClient

from app.core.config import settings


def test_create_and_read_rag_document(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    data = {
        "title": "Machine Learning Best Practices",
        "content": (
            "Deep learning models require careful hyperparameter tuning. "
            "Batch normalization and dropout prevent overfitting in dense neural networks."
        ),
        "content_type": "text/plain",
    }
    # Ingest document
    response = client.post(
        f"{settings.API_V1_STR}/rag/documents",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    created_doc = response.json()
    assert created_doc["title"] == data["title"]
    assert created_doc["chunk_count"] >= 1
    doc_id = created_doc["id"]

    # Read specific document
    get_res = client.get(
        f"{settings.API_V1_STR}/rag/documents/{doc_id}",
        headers=superuser_token_headers,
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # List documents
    list_res = client.get(
        f"{settings.API_V1_STR}/rag/documents",
        headers=superuser_token_headers,
    )
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["count"] >= 1
    assert any(d["id"] == doc_id for d in list_data["data"])

    # Search knowledge base
    search_res = client.post(
        f"{settings.API_V1_STR}/rag/search",
        headers=superuser_token_headers,
        json={"query": "hyperparameter tuning neural networks", "top_k": 3},
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total"] >= 1
    assert any(r["document_id"] == doc_id for r in search_data["results"])

    # Query knowledge base
    query_res = client.post(
        f"{settings.API_V1_STR}/rag/query",
        headers=superuser_token_headers,
        json={"query": "What prevents overfitting?", "top_k": 2},
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert len(query_data["answer"]) > 0
    assert len(query_data["sources"]) > 0

    # Delete document
    del_res = client.delete(
        f"{settings.API_V1_STR}/rag/documents/{doc_id}",
        headers=superuser_token_headers,
    )
    assert del_res.status_code == 200
    assert "deleted" in del_res.json()["message"]


def test_rag_document_not_found(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    non_existent_id = uuid.uuid4()
    res = client.get(
        f"{settings.API_V1_STR}/rag/documents/{non_existent_id}",
        headers=superuser_token_headers,
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Document not found"
