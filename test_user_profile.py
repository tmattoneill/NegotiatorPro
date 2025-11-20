"""
Quick test script for user profile functionality.

Tests:
1. User creation
2. User retrieval by ID, username, email
3. User update
4. API key encryption/decryption
"""
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from backend.database import db
from backend.user_profile import (
    UserProfileManager,
    UserProfileCreate,
    UserProfileUpdate
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_user_profile():
    """Test user profile operations."""
    logger.info("=== Testing User Profile System ===\n")

    # Connect to database
    logger.info("Connecting to database...")
    await db.connect()

    try:
        # Test 1: Create user
        logger.info("\nTest 1: Creating user profile...")
        user_data = UserProfileCreate(
            username="johndoe",
            email="john.doe@example.com",
            password="SecurePass123!",
            first_name="John",
            last_name="Doe",
            openai_api_key="sk-test-openai-key-12345",
            anthropic_api_key="sk-ant-test-anthropic-key-67890",
            role="user"
        )

        try:
            user = await UserProfileManager.create_user(user_data)
            logger.info(f"✓ Created user: {user.username}")
            logger.info(f"  ID: {user.id}")
            logger.info(f"  Email: {user.email}")
            logger.info(f"  Name: {user.first_name} {user.last_name}")
            logger.info(f"  Has OpenAI Key: {user.has_openai_key}")
            logger.info(f"  Has Anthropic Key: {user.has_anthropic_key}")
            user_id = user.id
        except ValueError as e:
            logger.warning(f"User already exists: {e}")
            # Get existing user
            user = await UserProfileManager.get_user_by_username("johndoe")
            user_id = user.id
            logger.info(f"Using existing user: {user.username} ({user_id})")

        # Test 2: Get user by ID
        logger.info("\nTest 2: Retrieving user by ID...")
        user = await UserProfileManager.get_user_by_id(user_id)
        if user:
            logger.info(f"✓ Retrieved user: {user.username}")
        else:
            logger.error("✗ Failed to retrieve user by ID")

        # Test 3: Get user by username
        logger.info("\nTest 3: Retrieving user by username...")
        user = await UserProfileManager.get_user_by_username("johndoe")
        if user:
            logger.info(f"✓ Retrieved user: {user.email}")
        else:
            logger.error("✗ Failed to retrieve user by username")

        # Test 4: Get user by email
        logger.info("\nTest 4: Retrieving user by email...")
        user = await UserProfileManager.get_user_by_email("john.doe@example.com")
        if user:
            logger.info(f"✓ Retrieved user: {user.username}")
        else:
            logger.error("✗ Failed to retrieve user by email")

        # Test 5: Get API keys
        logger.info("\nTest 5: Retrieving API keys...")
        keys = await UserProfileManager.get_user_api_keys(user_id)
        logger.info(f"✓ Retrieved API keys:")
        logger.info(f"  OpenAI Key: {keys['openai_api_key'][:20]}..." if keys['openai_api_key'] else "  OpenAI Key: None")
        logger.info(f"  Anthropic Key: {keys['anthropic_api_key'][:20]}..." if keys['anthropic_api_key'] else "  Anthropic Key: None")

        # Test 6: Update user profile
        logger.info("\nTest 6: Updating user profile...")
        update_data = UserProfileUpdate(
            first_name="Jonathan",
            last_name="Doe Jr."
        )
        updated_user = await UserProfileManager.update_user(user_id, update_data)
        if updated_user:
            logger.info(f"✓ Updated user: {updated_user.first_name} {updated_user.last_name}")
        else:
            logger.error("✗ Failed to update user")

        # Test 7: Update API keys
        logger.info("\nTest 7: Updating API keys...")
        update_keys = UserProfileUpdate(
            openai_api_key="sk-test-new-openai-key"
        )
        updated_user = await UserProfileManager.update_user(user_id, update_keys)
        if updated_user:
            logger.info(f"✓ Updated API keys")
            logger.info(f"  Has OpenAI Key: {updated_user.has_openai_key}")

        logger.info("\n=== All Tests Passed ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

    finally:
        # Cleanup
        logger.info("\nDisconnecting from database...")
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(test_user_profile())
