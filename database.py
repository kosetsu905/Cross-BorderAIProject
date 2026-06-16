import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_job_usage_columns()
    ensure_tool_cache_table()


def ensure_job_usage_columns() -> None:
    statements = [
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS usage_metrics JSONB",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS cache_key VARCHAR(64)",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS source_job_id VARCHAR(64)",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS completion_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE PRECISION",
        "CREATE INDEX IF NOT EXISTS ix_workflow_jobs_cache_key ON workflow_jobs (cache_key)",
        "CREATE INDEX IF NOT EXISTS ix_workflow_jobs_source_job_id ON workflow_jobs (source_job_id)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def ensure_tool_cache_table() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS tool_cache_entries (
            cache_key VARCHAR(64) PRIMARY KEY,
            tool_name VARCHAR(128) NOT NULL,
            tool_version VARCHAR(64) NOT NULL,
            value JSONB,
            metadata JSONB,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_tool_cache_entries_tool_name ON tool_cache_entries (tool_name)",
        "CREATE INDEX IF NOT EXISTS ix_tool_cache_entries_expires_at ON tool_cache_entries (expires_at)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
