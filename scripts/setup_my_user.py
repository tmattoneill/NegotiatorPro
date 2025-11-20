"""
Setup Your Personal User Profile

This script creates your personal user account with your premium API keys.
Run this once after initial setup.
"""
import asyncio
import logging
import sys
import getpass
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


async def setup_user():
    """Create or update your personal user profile."""
    logger.info("=== Personal User Profile Setup ===\n")

    # Connect to database
    await db.connect()

    try:
        # Get user details
        print("Let's set up your personal user profile.\n")

        username = input("Username [default: admin]: ").strip() or "admin"
        email = input("Email [default: admin@negotiatorpro.local]: ").strip() or "admin@negotiatorpro.local"

        # Check if user exists
        existing_user = await UserProfileManager.get_user_by_username(username)

        if existing_user:
            logger.info(f"User '{username}' already exists.")
            update = input("Do you want to update API keys? (y/n): ").strip().lower()

            if update == 'y':
                print("\nEnter your premium API keys (leave blank to skip):\n")

                openai_key = getpass.getpass("OpenAI API Key (sk-...): ").strip()
                anthropic_key = getpass.getpass("Anthropic API Key (sk-ant-...): ").strip()

                from backend.user_profile import UserProfileUpdate

                update_data = UserProfileUpdate()
                if openai_key:
                    update_data.openai_api_key = openai_key
                if anthropic_key:
                    update_data.anthropic_api_key = anthropic_key

                if openai_key or anthropic_key:
                    updated_user = await UserProfileManager.update_user(
                        existing_user.id,
                        update_data
                    )
                    logger.info("\n✓ API keys updated successfully!")
                    logger.info(f"  OpenAI key configured: {updated_user.has_openai_key}")
                    logger.info(f"  Anthropic key configured: {updated_user.has_anthropic_key}")
                else:
                    logger.info("No keys updated.")
            else:
                logger.info("Setup skipped.")

        else:
            # Create new user
            password = getpass.getpass("Password: ").strip()
            if not password:
                password = "admin123"
                logger.warning("Using default password: admin123")

            first_name = input("First name [optional]: ").strip() or None
            last_name = input("Last name [optional]: ").strip() or None

            print("\nEnter your premium API keys (leave blank to skip):\n")
            print("Note: These are encrypted before storage.")
            print("System uses Ollama Cloud for base models.\n")

            openai_key = getpass.getpass("OpenAI API Key (sk-...): ").strip() or None
            anthropic_key = getpass.getpass("Anthropic API Key (sk-ant-...): ").strip() or None

            user_data = UserProfileCreate(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                role="admin"
            )

            user = await UserProfileManager.create_user(user_data)

            logger.info("\n✓ User created successfully!")
            logger.info(f"  Username: {user.username}")
            logger.info(f"  Email: {user.email}")
            logger.info(f"  Role: {user.role}")
            logger.info(f"  OpenAI key configured: {user.has_openai_key}")
            logger.info(f"  Anthropic key configured: {user.has_anthropic_key}")

        print("\n" + "="*50)
        print("Setup Complete!")
        print("="*50)
        print("\nYour configuration:")
        print("  Base models: Ollama Cloud (system-level)")
        print("  Premium models: Your personal API keys")
        print("\nWhen you use premium models (GPT-4, Claude), your")
        print("personal API keys will be used and you'll be billed.")
        print("="*50)

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(setup_user())
