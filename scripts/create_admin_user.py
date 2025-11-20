#!/usr/bin/env python3
"""
Create default admin user in the database.

Usage:
    python scripts/create_admin_user.py

Or from Docker:
    docker compose exec backend python scripts/create_admin_user.py
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.user_profile import UserProfileManager, UserProfileCreate
from backend.database import db


async def create_admin_user():
    """Create the default admin user."""
    try:
        # Connect to database
        await db.connect()
        print("✓ Connected to database")

        # Check if admin user already exists
        existing = await UserProfileManager.get_user_by_username("admin")
        if existing:
            print(f"✓ Admin user already exists (ID: {existing.id})")
            print(f"  Username: {existing.username}")
            print(f"  Email: {existing.email}")
            return

        # Create admin user
        admin_profile = UserProfileCreate(
            username="admin",
            email="test@example.com",
            password="admin123",
            first_name="Admin",
            last_name="User",
            role="admin"
        )

        user = await UserProfileManager.create_user(admin_profile)

        print("✅ Admin user created successfully!")
        print(f"  ID: {user.id}")
        print(f"  Username: {user.username}")
        print(f"  Email: {user.email}")
        print(f"  Role: {user.role}")
        print(f"\n  Login credentials:")
        print(f"    Username: admin")
        print(f"    Password: admin123")

    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        raise
    finally:
        # Disconnect from database
        await db.disconnect()
        print("\n✓ Database connection closed")


if __name__ == "__main__":
    asyncio.run(create_admin_user())
