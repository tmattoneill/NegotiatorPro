"""
Docker-based User Profile Initialization Script

This script runs inside the Docker container to:
1. Wait for PostgreSQL to be ready
2. Create a default admin user if none exists
3. Set up initial configuration

This should be run as part of the Docker startup process.
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import db
from backend.user_profile import UserProfileManager, UserProfileCreate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def wait_for_database(max_retries=30, delay=2):
    """
    Wait for PostgreSQL to be ready.

    Args:
        max_retries: Maximum number of connection attempts
        delay: Delay between retries in seconds

    Returns:
        True if database is ready, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{max_retries})...")
            await db.connect()

            # Test connection
            healthy = await db.health_check()
            if healthy:
                logger.info("✓ Database connection established")
                return True

            await db.disconnect()

        except Exception as e:
            logger.warning(f"Database not ready: {e}")

        if attempt < max_retries - 1:
            logger.info(f"Waiting {delay} seconds before retry...")
            await asyncio.sleep(delay)

    logger.error("Failed to connect to database after maximum retries")
    return False


async def create_default_admin():
    """
    Create default admin user if none exists.

    Creates:
    - Username: admin
    - Email: admin@negotiatorpro.local
    - Password: admin123
    - Role: admin
    """
    try:
        # Check if admin user already exists
        existing_admin = await UserProfileManager.get_user_by_username("admin")

        if existing_admin:
            logger.info("✓ Admin user already exists")
            logger.info(f"  Username: {existing_admin.username}")
            logger.info(f"  Email: {existing_admin.email}")
            return existing_admin

        # Create admin user
        admin_data = UserProfileCreate(
            username="admin",
            email="admin@negotiatorpro.local",
            password="admin123",
            first_name="System",
            last_name="Administrator",
            role="admin"
        )

        admin_user = await UserProfileManager.create_user(admin_data)
        logger.info("✓ Created default admin user")
        logger.info(f"  Username: admin")
        logger.info(f"  Email: admin@negotiatorpro.local")
        logger.info(f"  Password: admin123")
        logger.info(f"  User ID: {admin_user.id}")
        logger.warning("⚠️  IMPORTANT: Change the admin password after first login!")

        return admin_user

    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        return None


async def create_test_user():
    """
    Create a test user for development/testing.

    Creates:
    - Username: testuser
    - Email: test@example.com
    - Password: testpass123
    - Role: user
    """
    try:
        # Check if test user already exists
        existing_user = await UserProfileManager.get_user_by_username("testuser")

        if existing_user:
            logger.info("✓ Test user already exists")
            return existing_user

        # Create test user
        test_data = UserProfileCreate(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            first_name="Test",
            last_name="User",
            role="user"
        )

        test_user = await UserProfileManager.create_user(test_data)
        logger.info("✓ Created test user")
        logger.info(f"  Username: testuser")
        logger.info(f"  Email: test@example.com")
        logger.info(f"  Password: testpass123")

        return test_user

    except Exception as e:
        logger.error(f"Failed to create test user: {e}")
        return None


async def main():
    """Main initialization function."""
    logger.info("=== NegotiatorPro User Profile Initialization ===\n")

    # Step 1: Wait for database
    logger.info("Step 1: Waiting for PostgreSQL to be ready...")
    db_ready = await wait_for_database()

    if not db_ready:
        logger.error("Database is not available. Exiting.")
        sys.exit(1)

    # Step 2: Create default admin user
    logger.info("\nStep 2: Creating default admin user...")
    admin = await create_default_admin()

    if not admin:
        logger.error("Failed to create admin user. Exiting.")
        await db.disconnect()
        sys.exit(1)

    # Step 3: Create test user (optional, for development)
    logger.info("\nStep 3: Creating test user...")
    await create_test_user()

    # Cleanup
    logger.info("\nDisconnecting from database...")
    await db.disconnect()

    logger.info("\n=== Initialization Complete ===")
    logger.info("\nDefault users created:")
    logger.info("1. Admin: admin / admin123")
    logger.info("2. Test User: testuser / testpass123")
    logger.info("\nAPI available at: http://localhost:8000/api/docs")


if __name__ == "__main__":
    asyncio.run(main())
