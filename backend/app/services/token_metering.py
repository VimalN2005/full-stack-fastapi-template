import uuid
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlmodel import Session, col, func, select

from app.models import TokenUsage, User

# Model pricing in USD per 1,000,000 tokens (Standard OpenAI/Anthropic rates)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt": 0.15,
        "completion": 0.60,
    },
    "text-embedding-3-small": {
        "prompt": 0.02,
        "completion": 0.00,
    },
    "default": {
        "prompt": 0.20,
        "completion": 0.80,
    },
}


def calculate_token_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate the estimated USD cost of an AI inference request based on model pricing."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
    return round(prompt_cost + completion_cost, 6)


def record_token_usage(
    session: Session,
    user_id: uuid.UUID,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> TokenUsage:
    """Record token consumption and estimated cost for a user in PostgreSQL."""
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_token_cost(model_name, prompt_tokens, completion_tokens)

    usage = TokenUsage(
        user_id=user_id,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
    )
    session.add(usage)
    session.commit()
    session.refresh(usage)
    return usage


def get_current_month_usage(
    session: Session,
    user_id: uuid.UUID,
) -> tuple[int, float]:
    """Calculate the total tokens consumed and estimated spend by a user in the current calendar month (UTC)."""
    now = datetime.now(UTC)
    start_of_month = datetime(now.year, now.month, 1, tzinfo=UTC)

    statement = (
        select(
            func.coalesce(func.sum(TokenUsage.total_tokens), 0),
            func.coalesce(func.sum(TokenUsage.estimated_cost_usd), 0.0),
        )
        .where(TokenUsage.user_id == user_id)
        .where(col(TokenUsage.created_at) >= start_of_month)
    )
    total_tokens, total_cost = session.exec(statement).one()
    return int(total_tokens), float(total_cost)


def check_token_quota(
    session: Session,
    user: User,
    estimated_tokens: int = 100,
) -> None:
    """Enforce user token quota. Superusers are exempt.

    Raises HTTP 429 Too Many Requests if the user's monthly quota is exhausted.
    """
    if user.is_superuser:
        return

    used_tokens, _ = get_current_month_usage(session, user.id)
    limit = user.monthly_token_limit

    if used_tokens + estimated_tokens > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly AI token quota exceeded (Used: {used_tokens:,} / "
                f"Limit: {limit:,} tokens). Please upgrade your quota."
            ),
        )
