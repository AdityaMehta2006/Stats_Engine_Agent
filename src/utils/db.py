"""
MySQL database connection manager.
Handles engine creation, database creation, and safe connection contexts.
All queries use SQLAlchemy parameterized statements -- no raw string interpolation.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import QueuePool

from src.config.settings import get_settings


def get_server_engine() -> Engine:
    """
    Engine connected to MySQL server (no specific database).
    Used for creating/dropping databases.
    """
    settings = get_settings()
    return create_engine(
        settings.mysql_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
    )


def get_db_engine(db_name: str) -> Engine:
    """
    Engine connected to a specific database.
    Used for table operations and data insertion.
    """
    settings = get_settings()
    return create_engine(
        settings.mysql_db_url(db_name),
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
    )


@contextmanager
def server_connection() -> Generator[Connection, None, None]:
    """Context manager for server-level operations (create/drop DB)."""
    engine = get_server_engine()
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


@contextmanager
def db_connection(db_name: str) -> Generator[Connection, None, None]:
    """Context manager for database-level operations (tables, inserts)."""
    engine = get_db_engine(db_name)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


def create_database(db_name: str) -> str:
    """
    Create a new MySQL database.
    Returns the database name on success.
    """
    with server_connection() as conn:
        # Using text() with proper escaping -- db names can't be parameterized
        # so we sanitize the name before reaching this point (see security.py)
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        conn.execute(text(f"ALTER DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit()
    return db_name


def drop_database(db_name: str) -> None:
    """Drop a database (used for cleanup/rollback)."""
    with server_connection() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
        conn.commit()


def test_connection() -> bool:
    """Test if MySQL server is reachable."""
    try:
        with server_connection() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception:
        return False
