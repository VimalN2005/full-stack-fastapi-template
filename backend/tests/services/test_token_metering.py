import uuid

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.models import User
from app.services.token_metering import (
    calculate_token_cost,
    check_token_quota,
    get_current_month_usage,
    record_token_usage,
)


def test_calculate_token_cost() -> None:
    # 1,000,000 prompt tokens of gpt-4o-mini = $0.15
    cost = calculate_token_cost(
        "gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert cost == 0.15

    # 1,000,000 completion tokens of gpt-4o-mini = $0.60
    cost_comp = calculate_token_cost(
        "gpt-4o-mini", prompt_tokens=0, completion_tokens=1_000_000
    )
    assert cost_comp == 0.60

    # embedding: 1,000,000 tokens = $0.02
    cost_embed = calculate_token_cost(
        "text-embedding-3-small", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert cost_embed == 0.02


def test_record_and_get_monthly_usage(db: Session) -> None:
    test_user_id = uuid.uuid4()
    # Record usage 1
    usage1 = record_token_usage(
        session=db,
        user_id=test_user_id,
        model_name="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert usage1.total_tokens == 1500
    assert usage1.user_id == test_user_id
    assert usage1.estimated_cost_usd > 0

    # Record usage 2
    record_token_usage(
        session=db,
        user_id=test_user_id,
        model_name="text-embedding-3-small",
        prompt_tokens=500,
        completion_tokens=0,
    )

    total_tokens, total_cost = get_current_month_usage(db, test_user_id)
    assert total_tokens == 2000
    assert total_cost > 0


def test_check_token_quota_within_limit(db: Session) -> None:
    user = User(
        id=uuid.uuid4(),
        email=f"quota_ok_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="fake",
        monthly_token_limit=10000,
        is_superuser=False,
    )
    db.add(user)
    db.commit()

    # Should not raise exception
    check_token_quota(db, user, estimated_tokens=100)


def test_check_token_quota_exceeded_raises_429(db: Session) -> None:
    user = User(
        id=uuid.uuid4(),
        email=f"quota_exceeded_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="fake",
        monthly_token_limit=1000,
        is_superuser=False,
    )
    db.add(user)
    db.commit()

    # Consume all tokens
    record_token_usage(
        session=db,
        user_id=user.id,
        model_name="gpt-4o-mini",
        prompt_tokens=800,
        completion_tokens=300,
    )

    with pytest.raises(HTTPException) as exc_info:
        check_token_quota(db, user, estimated_tokens=100)

    assert exc_info.value.status_code == 429
    assert "Monthly AI token quota exceeded" in exc_info.value.detail


def test_check_token_quota_superuser_exempt(db: Session) -> None:
    superuser = User(
        id=uuid.uuid4(),
        email=f"admin_quota_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="fake",
        monthly_token_limit=100,  # low limit
        is_superuser=True,
    )
    db.add(superuser)
    db.commit()

    # Consume more than limit
    record_token_usage(
        session=db,
        user_id=superuser.id,
        model_name="gpt-4o-mini",
        prompt_tokens=5000,
        completion_tokens=5000,
    )

    # Superuser should never raise 429
    check_token_quota(db, superuser, estimated_tokens=1000)
