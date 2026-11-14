"""
ASTRAEUS Relational Storage Engine
Manages database connections, connection pooling, schema migrations,
and ACID transaction boundaries for PostgreSQL and SQLite.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Generator
from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.schema import CreateTable, CreateIndex

from astraeus_data.storage.schema import Base


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enforce foreign key constraints on SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_database_engine(url: str = "sqlite:///:memory:", echo: bool = False) -> Engine:
    """
    Creates and configures a SQLAlchemy database engine.
    Supports PostgreSQL (production) and SQLite (testing/local in-memory).
    """
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine = create_engine(url, echo=echo, connect_args=connect_args)
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    else:
        # PostgreSQL / other production backends
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
    
    # Tables in dependency order
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
