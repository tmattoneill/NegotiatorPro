"""
Database Connection Management

Provides PostgreSQL database connection pooling and session management
using asyncpg for high-performance async operations.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import asyncpg
from dotenv import load_dotenv

from .config_loader import config

# Load environment variables — project .env takes precedence over shell env
# so this project's keys/config are self-contained and don't depend on the
# developer's personal shell setup.
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class Database:
    """
    Database connection manager using asyncpg connection pool.

    Provides:
    - Connection pooling for efficient resource usage
    - Async context manager for safe connection handling
    - Automatic connection retry and error handling
    """

    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "negotiatorpro")
        self.user = os.getenv("POSTGRES_USER", "negotiatorpro")
        self.password = os.getenv("POSTGRES_PASSWORD", "")

        # Parse DATABASE_URL if provided (overrides individual settings)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            self._parse_database_url(database_url)

        self._pool: Optional[asyncpg.Pool] = None

    def _parse_database_url(self, url: str):
        """
        Parse DATABASE_URL in the format:
        postgresql://user:password@host:port/database
        """
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "")

            # Extract credentials and connection info
            if "@" in url:
                credentials, connection = url.split("@", 1)
                if ":" in credentials:
                    self.user, self.password = credentials.split(":", 1)
                else:
                    self.user = credentials

                # Extract host, port, database
                if "/" in connection:
                    host_port, self.database = connection.split("/", 1)
                    if ":" in host_port:
                        self.host, port_str = host_port.split(":", 1)
                        self.port = int(port_str)
                    else:
                        self.host = host_port

    async def connect(self) -> None:
        """
        Create connection pool.

        Raises:
            asyncpg.PostgresError: If connection fails
        """
        if self._pool is None:
            try:
                logger.info(f"Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}")
                # Get pool settings from config
                pool_min = config.get("database.pool_min_size", 2)
                pool_max = config.get("database.pool_max_size", 10)
                cmd_timeout = config.get("database.command_timeout_seconds", 60)

                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=pool_min,
                    max_size=pool_max,
                    command_timeout=cmd_timeout,
                )
                logger.info("Database connection pool created successfully")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Acquire a database connection from the pool.

        Usage:
            async with db.acquire() as conn:
                result = await conn.fetch("SELECT * FROM users")

        Yields:
            asyncpg.Connection: Database connection
        """
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query without returning results.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Status message from query execution
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """
        Fetch all rows from a query.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of asyncpg.Record objects
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Fetch a single row from a query.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            asyncpg.Record or None if no results
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """
        Fetch a single value from a query.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single value or None
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def health_check(self) -> bool:
        """
        Check if database connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            async with self.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
db = Database()
