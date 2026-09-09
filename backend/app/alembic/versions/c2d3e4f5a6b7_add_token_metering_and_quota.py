"""Add token metering and user monthly quota

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-10 00:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'c2d3e4f5a6b7'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add monthly_token_limit column to user table with default 50000
    op.add_column(
        'user',
        sa.Column('monthly_token_limit', sa.Integer(), nullable=False, server_default='50000')
    )

    # 2. Create tokenusage table
    op.create_table(
        'tokenusage',
        sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('completion_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. Create indexes for fast monthly aggregation by user
    op.create_index(op.f('ix_tokenusage_user_id'), 'tokenusage', ['user_id'], unique=False)
    op.create_index(op.f('ix_tokenusage_created_at'), 'tokenusage', ['created_at'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_tokenusage_created_at'), table_name='tokenusage')
    op.drop_index(op.f('ix_tokenusage_user_id'), table_name='tokenusage')
    op.drop_table('tokenusage')
    op.drop_column('user', 'monthly_token_limit')
