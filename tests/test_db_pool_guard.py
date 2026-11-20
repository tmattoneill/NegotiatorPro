"""
Guards for the Neon idle rule (docs/deployment/DEPLOY.md).

Neon bills compute for as long as any connection is open. In November 2026 the
backend opened pool connections at boot and held them forever, which kept the
dev and prod computes awake 24/7 and burned the monthly allowance in eight
days. These tests fail if that regresses in config or in startup code.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEON_SUSPEND_SECONDS = 300


@pytest.mark.unit
def test_config_pool_can_drain_to_zero():
    with open(PROJECT_ROOT / "config.json") as handle:
        database = json.load(handle)["database"]

    assert database["pool_min_size"] == 0, (
        "pool_min_size must be 0 or the pool holds connections at idle and Neon never suspends"
    )
    assert 0 < database["idle_connection_lifetime_seconds"] <= NEON_SUSPEND_SECONDS, (
        "idle connections must close well inside Neon's five-minute suspend timer"
    )


@pytest.mark.unit
def test_create_pool_receives_lazy_settings():
    from backend.database import Database

    with patch("backend.database.asyncpg.create_pool", new_callable=AsyncMock) as create_pool:
        create_pool.return_value = object()
        asyncio.run(Database().connect())

    kwargs = create_pool.call_args.kwargs
    assert kwargs["min_size"] == 0
    assert 0 < kwargs["max_inactive_connection_lifetime"] <= NEON_SUSPEND_SECONDS


@pytest.mark.unit
def test_lifespan_does_not_connect_to_database():
    from backend.api import main

    async def run_startup():
        async with main.lifespan(main.app):
            pass

    with patch.object(main, "_hydrate_from_storage"), \
         patch("backend.utils.env_validator.validate_startup_environment"), \
         patch("backend.database.db.connect", new_callable=AsyncMock) as connect, \
         patch("backend.database.db.disconnect", new_callable=AsyncMock):
        asyncio.run(run_startup())

    connect.assert_not_called()
