"""Add chat sessions and messages for multi-turn RAG memory

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-10 01:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'd3e4f5a6b7c8'
down_revision = 'c2d3e4f5a6b7'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Create chatsession table
    op.create_table(
        'chatsession',
        sa.Column('title', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chatsession_user_id'), 'chatsession', ['user_id'], unique=False)
    op.create_index(op.f('ix_chatsession_created_at'), 'chatsession', ['created_at'], unique=False)
    op.create_index(op.f('ix_chatsession_updated_at'), 'chatsession', ['updated_at'], unique=False)

    # 2. Create chatmessage table
    op.create_table(
        'chatmessage',
        sa.Column('role', sqlmodel.sql.sqltypes.AutoString(length=20), nullable=False),
        sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('sources', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('session_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['chatsession.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chatmessage_session_id'), 'chatmessage', ['session_id'], unique=False)
    op.create_index(op.f('ix_chatmessage_created_at'), 'chatmessage', ['created_at'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_chatmessage_created_at'), table_name='chatmessage')
    op.drop_index(op.f('ix_chatmessage_session_id'), table_name='chatmessage')
    op.drop_table('chatmessage')

    op.drop_index(op.f('ix_chatsession_updated_at'), table_name='chatsession')
    op.drop_index(op.f('ix_chatsession_created_at'), table_name='chatsession')
    op.drop_index(op.f('ix_chatsession_user_id'), table_name='chatsession')
    op.drop_table('chatsession')
