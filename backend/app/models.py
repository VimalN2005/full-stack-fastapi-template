import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from pydantic import EmailStr
from sqlalchemy import Column, DateTime, Text
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)
    monthly_token_limit: int = Field(default=50000)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)
    monthly_token_limit: int | None = None


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    items: list[Item] = Relationship(back_populates="owner", cascade_delete=True)
    documents: list[Document] = Relationship(
        back_populates="owner", cascade_delete=True
    )
    token_usages: list[TokenUsage] = Relationship(
        back_populates="user", cascade_delete=True
    )
    chat_sessions: list[ChatSession] = Relationship(
        back_populates="user", cascade_delete=True
    )


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(SQLModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


# ==========================================
# RAG (Retrieval-Augmented Generation) Models
# ==========================================


class DocumentBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="text/plain", max_length=50)


class DocumentCreate(DocumentBase):
    content: str = Field(min_length=1)


class Document(DocumentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="documents")
    chunks: list[DocumentChunk] = Relationship(
        back_populates="document", cascade_delete=True
    )


class DocumentPublic(DocumentBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None
    chunk_count: int = 0


class DocumentsPublic(SQLModel):
    data: list[DocumentPublic]
    count: int


class DocumentChunk(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        foreign_key="document.id", nullable=False, ondelete="CASCADE"
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, index=True, ondelete="CASCADE"
    )
    chunk_index: int = Field(default=0)
    content: str = Field(sa_column=Column(Text, nullable=False))
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(1536), nullable=True),
    )
    document: Document | None = Relationship(back_populates="chunks")


# RAG Search & Query schemas
class RAGSearchRequest(SQLModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank: bool = Field(default=True)


class RAGChunkMatch(SQLModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    score: float
    match_type: str = "hybrid"  # "dense", "keyword", "hybrid", or "reranked"


class RAGSearchResponse(SQLModel):
    query: str
    results: list[RAGChunkMatch]
    total: int


class RAGQueryRequest(SQLModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)


class RAGQueryResponse(SQLModel):
    query: str
    answer: str
    sources: list[RAGChunkMatch]


# ==========================================
# AI Token Metering & Quota Models
# ==========================================


class TokenUsageBase(SQLModel):
    model_name: str = Field(max_length=100)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)


class TokenUsage(TokenUsageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, index=True, ondelete="CASCADE"
    )
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    user: User | None = Relationship(back_populates="token_usages")


class TokenUsagePublic(TokenUsageBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime | None = None


class TokenUsagesPublic(SQLModel):
    data: list[TokenUsagePublic]
    count: int


class AIUsageStatsResponse(SQLModel):
    total_tokens_month: int
    monthly_limit: int
    remaining_tokens: int
    estimated_cost_usd: float
    usage_percentage: float
    is_unlimited: bool


class UpdateQuotaRequest(SQLModel):
    monthly_token_limit: int = Field(ge=0)


# ==========================================
# Chat Session & Multi-Turn Memory Models
# ==========================================


class ChatSessionBase(SQLModel):
    title: str = Field(default="New Chat", max_length=255)


class ChatSessionCreate(SQLModel):
    title: str | None = Field(default=None, max_length=255)


class ChatSession(ChatSessionBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, index=True, ondelete="CASCADE"
    )
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    updated_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    user: User | None = Relationship(back_populates="chat_sessions")
    messages: list[ChatMessage] = Relationship(
        back_populates="session", cascade_delete=True
    )


class ChatMessageBase(SQLModel):
    role: str = Field(max_length=20)  # "user", "assistant", or "system"
    content: str


class ChatMessageCreate(SQLModel):
    content: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)


class ChatMessage(ChatMessageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(
        foreign_key="chatsession.id", nullable=False, index=True, ondelete="CASCADE"
    )
    sources: str | None = Field(default=None)  # JSON-serialized list of sources
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    session: ChatSession | None = Relationship(back_populates="messages")


class ChatMessagePublic(ChatMessageBase):
    id: uuid.UUID
    session_id: uuid.UUID
    sources: str | None = None
    created_at: datetime | None = None


class ChatSessionPublic(ChatSessionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None
    message_count: int = 0


class ChatSessionsPublic(SQLModel):
    data: list[ChatSessionPublic]
    count: int


class ChatSessionDetailPublic(ChatSessionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[ChatMessagePublic] = []
