"""Add pgvector extension and RAG models

Revision ID: b1c2d3e4f5a6
Revises: fe56fa70289e
Create Date: 2026-09-09 17:30:00.000000

"""
from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = 'fe56fa70289e'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create document table
    op.create_table(
        'document',
        sa.Column('title', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('content_type', sqlmodel.sql.sqltypes.AutoString(length=50), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('owner_id', sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. Create documentchunk table with pgvector column
    op.create_table(
        'documentchunk',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column('owner_id', sa.Uuid(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['document.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. Create indexes for high performance
    op.create_index(op.f('ix_documentchunk_owner_id'), 'documentchunk', ['owner_id'], unique=False)
    op.create_index(op.f('ix_documentchunk_document_id'), 'documentchunk', ['document_id'], unique=False)

    # 5. Create HNSW index for vector cosine similarity search
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documentchunk_embedding_hnsw "
        "ON documentchunk USING hnsw (embedding vector_cosine_ops);"
    )

    # 6. Create Full-Text Search GIN index for hybrid keyword search
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documentchunk_content_fts "
        "ON documentchunk USING gin (to_tsvector('english', content));"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_documentchunk_content_fts;")
    op.execute("DROP INDEX IF EXISTS ix_documentchunk_embedding_hnsw;")
    op.drop_index(op.f('ix_documentchunk_document_id'), table_name='documentchunk')
    op.drop_index(op.f('ix_documentchunk_owner_id'), table_name='documentchunk')
    op.drop_table('documentchunk')
    op.drop_table('document')
