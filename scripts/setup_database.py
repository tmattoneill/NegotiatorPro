"""
Database Setup and Migration Script

This script:
1. Creates the PostgreSQL database if it doesn't exist
2. Runs all migration scripts in order
3. Creates a test user profile
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "negotiatorpro")
    user = os.getenv("POSTGRES_USER", "negotiatorpro")
    password = os.getenv("POSTGRES_PASSWORD", "")

    try:
        # Connect to postgres database to create our database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=password
        )

        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            database
        )

        if not exists:
            # Create database
            await conn.execute(f'CREATE DATABASE {database}')
            logger.info(f"Created database: {database}")
        else:
            logger.info(f"Database already exists: {database}")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        logger.info("Please create the database manually or check your PostgreSQL connection")
        return False


async def run_migration(conn, migration_file: Path):
    """Run a single migration file."""
    logger.info(f"Running migration: {migration_file.name}")

    with open(migration_file, 'r') as f:
        sql = f.read()

    try:
        await conn.execute(sql)
        logger.info(f"✓ Migration completed: {migration_file.name}")
        return True
    except Exception as e:
        logger.error(f"✗ Migration failed: {migration_file.name}")
        logger.error(f"Error: {e}")
        return False


async def run_migrations():
    """Run all migration scripts."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "negotiatorpro")
    user = os.getenv("POSTGRES_USER", "negotiatorpro")
    password = os.getenv("POSTGRES_PASSWORD", "")

    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        # Find all migration files
        migrations_dir = Path(__file__).parent.parent / "migrations"
        migration_files = sorted(migrations_dir.glob("*.sql"))

        if not migration_files:
            logger.warning("No migration files found")
            return True

        logger.info(f"Found {len(migration_files)} migration(s)")

        # Run each migration
        for migration_file in migration_files:
            success = await run_migration(conn, migration_file)
            if not success:
                logger.error("Migration failed. Stopping.")
                await conn.close()
                return False

        await conn.close()
        logger.info("All migrations completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        return False


async def create_test_user():
    """Create a test user profile."""
    from backend.user_profile import UserProfileManager, UserProfileCreate

    try:
        # Create test user
        test_user = UserProfileCreate(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            first_name="Test",
            last_name="User",
            role="user"
        )

        user = await UserProfileManager.create_user(test_user)
        logger.info(f"✓ Created test user: {user.username} ({user.email})")
        logger.info(f"  User ID: {user.id}")
        logger.info(f"  Role: {user.role}")
        return True

    except ValueError as e:
        logger.warning(f"Test user already exists: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to create test user: {e}")
        return False


def generate_encryption_key():
    """Generate and display encryption key."""
    encryption_key = os.getenv("ENCRYPTION_KEY")

    if not encryption_key:
        logger.warning("No ENCRYPTION_KEY found in .env file")
        logger.info("Generating new encryption key...")

        key = Fernet.generate_key().decode()
        logger.info(f"\nAdd this to your .env file:")
        logger.info(f"ENCRYPTION_KEY={key}\n")
        return key
    else:
        logger.info("✓ ENCRYPTION_KEY is configured")
        return encryption_key


async def main():
    """Main setup function."""
    logger.info("=== NegotiatorPro Database Setup ===\n")

    # Step 1: Generate encryption key if needed
    logger.info("Step 1: Checking encryption key...")
    generate_encryption_key()

    # Step 2: Create database
    logger.info("\nStep 2: Creating database...")
    if not await create_database_if_not_exists():
        logger.error("Failed to create database. Exiting.")
        return

    # Step 3: Run migrations
    logger.info("\nStep 3: Running migrations...")
    if not await run_migrations():
        logger.error("Failed to run migrations. Exiting.")
        return

    # Step 4: Create test user
    logger.info("\nStep 4: Creating test user...")
    from backend.database import db
    await db.connect()

    await create_test_user()

    await db.disconnect()

    logger.info("\n=== Setup Complete ===")
    logger.info("\nYou can now:")
    logger.info("1. Start the API: ./run-api.sh")
    logger.info("2. Test user endpoints at: http://localhost:8000/api/docs")
    logger.info("3. Login with: testuser / testpass123")


if __name__ == "__main__":
    asyncio.run(main())
