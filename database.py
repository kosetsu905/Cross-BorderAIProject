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


def ensure_job_usage_columns() -> None:
    statements = [
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS usage_metrics JSONB",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS completion_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION",
        "ALTER TABLE workflow_jobs ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE PRECISION",
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
