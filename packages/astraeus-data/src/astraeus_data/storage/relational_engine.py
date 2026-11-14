"""
ASTRAEUS Relational Storage Engine
Manages database connections, connection pooling, schema migrations,
and ACID transaction boundaries for PostgreSQL and SQLite.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Generator, Optional
from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.schema import CreateTable, CreateIndex

from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode
from astraeus_data.storage.schema import Base


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enforce foreign key constraints on SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_database_engine(
    url: str = "sqlite:///:memory:",
    echo: bool = False,
    backend_mode: BackendMode = BackendMode.LOCAL_EMBEDDED,
) -> Engine:
    """
    Creates and configures a SQLAlchemy database engine.
    Supports PostgreSQL (production) and SQLite (testing/local in-memory).
    Prohibits silent fallback when BackendMode.STRICT_LIVE is enabled.
    """
    if backend_mode == BackendMode.STRICT_LIVE:
        if not url.startswith("postgresql"):
            raise LiveBackendUnavailableError(
                f"STRICT_LIVE mode requires a production PostgreSQL URL (e.g. 'postgresql+psycopg2://...'), "
                f"got '{url}'. SQLite and silent fallback are strictly prohibited."
            )
        try:
            engine = create_engine(
                url,
                echo=echo,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as exc:
            raise LiveBackendUnavailableError(
                f"STRICT_LIVE mode: Failed to connect to PostgreSQL server at '{url}': {exc}. "
                "Silent fallback is prohibited."
            ) from exc

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine = create_engine(url, echo=echo, connect_args=connect_args)
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    else:
        engine = create_engine(
            url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return engine


def init_schema(engine: Engine) -> None:
    """Initializes all database tables, constraints, and indexes defined in Base."""
    Base.metadata.create_all(engine)


def drop_schema(engine: Engine) -> None:
    """Drops all tables defined in Base."""
    Base.metadata.drop_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Returns a configured sessionmaker for the engine."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextlib.contextmanager
def transactional_session(engine: Engine) -> Generator[Session, None, None]:
    """
    ACID transaction context manager.
    Automatically commits on normal exit and rolls back on exception.
    """
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def generate_postgresql_ddl() -> str:
    """
    Compiles and exports the complete PostgreSQL DDL statements
    for the ASTRAEUS 4-entity schema, indexes, and constraints.
    """
    from sqlalchemy.dialects import postgresql
    ddl_statements = []
    
    tables = [
        Base.metadata.tables["persistent_sky_loci"],
        Base.metadata.tables["physical_transient_episodes"],
        Base.metadata.tables["catalog_reports"],
        Base.metadata.tables["host_counterparts"],
        Base.metadata.tables["candidates_10k"],
        Base.metadata.tables["photometry_records"],
        Base.metadata.tables["raw_snapshots"],
    ]
    
    for table in tables:
        create_table = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        ddl_statements.append(create_table.strip() + ";\n")
        
        for idx in table.indexes:
            create_index = str(CreateIndex(idx).compile(dialect=postgresql.dialect()))
            ddl_statements.append(create_index.strip() + ";\n")

    return "\n".join(ddl_statements)
