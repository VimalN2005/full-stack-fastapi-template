import socket
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, delete

from app.api.deps import get_db
from app.core.config import settings
from app.core.db import engine, init_db
from app.main import app
from app.models import (
    ChatMessage,
    ChatSession,
    Document,
    DocumentChunk,
    Item,
    TokenUsage,
    User,
)
from tests.utils.user import authentication_token_from_email
from tests.utils.utils import get_superuser_token_headers


def _is_postgres_available() -> bool:
    try:
        url_str = str(settings.DATABASE_URL)
        host_port = url_str.split("@")[-1].split("/")[0]
        host = host_port.split(":")[0]
        port = int(host_port.split(":")[1]) if ":" in host_port else 5432
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except Exception:
        return False


if _is_postgres_available():
    test_engine = engine
else:
    from sqlalchemy.pool import StaticPool

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session]:
    def _get_test_db() -> Generator[Session]:
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_db] = _get_test_db

    with Session(test_engine) as session:
        init_db(session)
        yield session
        session.execute(delete(ChatMessage))
        session.execute(delete(ChatSession))
        session.execute(delete(TokenUsage))
        session.execute(delete(DocumentChunk))
        session.execute(delete(Document))
        session.execute(delete(Item))
        session.execute(delete(User))
        session.commit()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )
