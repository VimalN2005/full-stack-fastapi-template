"""Add status and error_message to document table

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-09-10 01:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'e4f5a6b7c8d9'
down_revision = 'd3e4f5a6b7c8'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'document',
        sa.Column('status', sqlmodel.sql.sqltypes.AutoString(length=20), nullable=False, server_default='ready')
    )
    op.add_column(
        'document',
        sa.Column('error_message', sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.create_index(op.f('ix_document_status'), 'document', ['status'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_document_status'), table_name='document')
    op.drop_column('document', 'error_message')
    op.drop_column('document', 'status')
