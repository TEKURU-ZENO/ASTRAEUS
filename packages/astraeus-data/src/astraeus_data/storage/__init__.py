"""
ASTRAEUS Storage Architecture & Persistence Models
"""

from enum import Enum


class BackendMode(str, Enum):
    """Execution profile for database and storage backends."""
    STRICT_LIVE = "strict_live"        # Fails immediately if live server is unreachable; NO silent fallback
    LOCAL_EMBEDDED = "local_embedded"  # In-memory / embedded engines for reproducible offline validation
