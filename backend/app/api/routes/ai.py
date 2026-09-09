import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    AIUsageStatsResponse,
    TokenUsage,
    TokenUsagePublic,
    TokenUsagesPublic,
    UpdateQuotaRequest,
    User,
    UserPublic,
)
from app.services.token_metering import get_current_month_usage

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/usage", response_model=AIUsageStatsResponse)
def get_ai_usage(
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Get the current authenticated user's AI token usage, spend, and quota statistics for the current month."""
    total_tokens, total_cost = get_current_month_usage(session, current_user.id)
    limit = current_user.monthly_token_limit

    if current_user.is_superuser:
        usage_pct = 0.0
        remaining = 999_999_999
    else:
        usage_pct = round((total_tokens / limit) * 100.0, 2) if limit > 0 else 0.0
        remaining = max(0, limit - total_tokens)

    return AIUsageStatsResponse(
        total_tokens_month=total_tokens,
        monthly_limit=limit,
        remaining_tokens=remaining,
        estimated_cost_usd=total_cost,
        usage_percentage=usage_pct,
        is_unlimited=current_user.is_superuser,
    )


@router.get("/history", response_model=TokenUsagesPublic)
def get_token_history(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 50,
    user_id: uuid.UUID | None = None,
) -> Any:
    """Get token consumption history for current user or all users (superuser only)."""
    target_user_id = current_user.id
    if user_id and current_user.is_superuser:
        target_user_id = user_id

    count_stmt = (
        select(func.count())
        .select_from(TokenUsage)
        .where(TokenUsage.user_id == target_user_id)
    )
    count = session.exec(count_stmt).one()

    stmt = (
        select(TokenUsage)
        .where(TokenUsage.user_id == target_user_id)
        .order_by(col(TokenUsage.created_at).desc())
        .offset(skip)
        .limit(limit)
    )
    usages = session.exec(stmt).all()

    data = [
        TokenUsagePublic(
            id=u.id,
            user_id=u.user_id,
            model_name=u.model_name,
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=u.estimated_cost_usd,
            created_at=u.created_at,
        )
        for u in usages
    ]
    return TokenUsagesPublic(data=data, count=count)


@router.patch("/users/{user_id}/quota", response_model=UserPublic)
def update_user_token_quota(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    user_id: uuid.UUID,
    quota_in: UpdateQuotaRequest,
) -> Any:
    """Update a user's monthly AI token limit (Superuser only)."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="The user doesn't have enough privileges"
        )

    target_user = session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    target_user.monthly_token_limit = quota_in.monthly_token_limit
    session.add(target_user)
    session.commit()
    session.refresh(target_user)
    return target_user
