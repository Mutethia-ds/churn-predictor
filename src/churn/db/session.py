"""Database engine/session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from churn.config import get_secrets
from churn.db.models import Base


def get_engine():
    secrets = get_secrets()
    return create_engine(secrets.database_url, future=True)


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    engine = get_engine()
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    engine = get_engine()
    session_factory = sessionmaker(bind=engine, future=True)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
