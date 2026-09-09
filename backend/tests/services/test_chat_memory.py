import uuid

from sqlmodel import Session, select

from app.models import ChatMessage, User
from app.services.chat_memory import (
    add_chat_message,
    create_chat_session,
    delete_chat_session,
    format_conversation_history,
    generate_multi_turn_answer,
    get_chat_session_with_messages,
)
from app.services.rag import ingest_document


def test_create_and_get_chat_session(db: Session) -> None:
    user_id = uuid.uuid4()
    session = create_chat_session(db, user_id=user_id, title="Custom Title")
    assert session.id is not None
    assert session.title == "Custom Title"
    assert session.user_id == user_id

    fetched = get_chat_session_with_messages(db, session.id, user_id)
    assert fetched is not None
    assert fetched.id == session.id


def test_add_chat_message_and_title_auto_update(db: Session) -> None:
    user_id = uuid.uuid4()
    session = create_chat_session(db, user_id=user_id)  # Default title "New Chat"
    assert session.title == "New Chat"

    msg = add_chat_message(
        db,
        session.id,
        role="user",
        content="How do transformers work in deep learning?",
    )
    assert msg.id is not None
    assert msg.role == "user"

    # Title should have been auto-updated from the first user prompt
    db.refresh(session)
    assert session.title == "How do transformers work in deep learning?"


def test_format_conversation_history() -> None:
    session_id = uuid.uuid4()
    msgs = [
        ChatMessage(session_id=session_id, role="user", content="Hi"),
        ChatMessage(
            session_id=session_id, role="assistant", content="Hello! How can I help?"
        ),
        ChatMessage(session_id=session_id, role="user", content="What is RAG?"),
        ChatMessage(
            session_id=session_id,
            role="assistant",
            content="Retrieval-Augmented Generation.",
        ),
    ]
    history = format_conversation_history(msgs, max_turns=2)
    assert "User: Hi" in history
    assert "Assistant: Hello! How can I help?" in history
    assert "User: What is RAG?" in history
    assert "Assistant: Retrieval-Augmented Generation." in history


def test_delete_chat_session_cascade(db: Session) -> None:
    user_id = uuid.uuid4()
    session = create_chat_session(db, user_id=user_id)
    add_chat_message(db, session.id, role="user", content="Message to be deleted")

    # Verify message exists
    msgs = db.exec(
        select(ChatMessage).where(ChatMessage.session_id == session.id)
    ).all()
    assert len(msgs) == 1

    # Delete session
    success = delete_chat_session(db, session.id, user_id)
    assert success is True

    # Verify messages were cascade-deleted
    msgs_after = db.exec(
        select(ChatMessage).where(ChatMessage.session_id == session.id)
    ).all()
    assert len(msgs_after) == 0


def test_generate_multi_turn_answer_flow(db: Session) -> None:
    user = User(
        id=uuid.uuid4(),
        email=f"multiturn_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="fake",
        monthly_token_limit=100000,
        is_superuser=False,
    )
    db.add(user)
    db.commit()

    # Ingest knowledge base doc
    ingest_document(
        db,
        user_id=user.id,
        title="PyTorch Neural Networks",
        content="PyTorch uses autograd for automatic differentiation. Tensors can be moved to CUDA devices.",
    )

    # Create session
    chat_session = create_chat_session(db, user_id=user.id)

    # Turn 1
    answer1, sources1, u1, a1 = generate_multi_turn_answer(
        session=db,
        user=user,
        session_id=chat_session.id,
        user_query="What does PyTorch use for automatic differentiation?",
    )
    assert len(answer1) > 0
    assert len(sources1) >= 1
    assert u1.role == "user"
    assert a1.role == "assistant"

    # Turn 2: Follow-up question relying on context
    answer2, sources2, u2, a2 = generate_multi_turn_answer(
        session=db,
        user=user,
        session_id=chat_session.id,
        user_query="Can tensors run on CUDA?",
    )
    assert len(answer2) > 0
    assert len(sources2) >= 1

    # Check that session now has 4 messages in total
    db.refresh(chat_session)
    assert len(chat_session.messages) == 4
