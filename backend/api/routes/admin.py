"""
Admin API Routes

Provides admin-only endpoints for system configuration.
Only accessible to users with role='admin'.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
import logging

from backend.prompt_manager import PromptManager
from backend.user_profile import UserProfileManager, UserProfile
from backend.database import db
from ..middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Initialize prompt manager
prompt_manager = PromptManager()

# Backup directory for system prompts
BACKUP_DIR = "data/system_prompt_backups"


class SystemPromptRequest(BaseModel):
    content: str


class SystemPromptResponse(BaseModel):
    content: str
    last_modified: Optional[str] = None


class BackupInfo(BaseModel):
    filename: str
    timestamp: str
    size: int


class UserListItem(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool


class NegotiationListItem(BaseModel):
    id: str
    user_id: str
    username: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


class UsageStats(BaseModel):
    user_id: str
    username: str
    total_requests: int
    total_tokens: int
    total_cost: float
    models_used: List[str]
    last_activity: Optional[datetime] = None


class DatabaseResetResponse(BaseModel):
    success: bool
    users_deleted: int
    negotiations_deleted: int
    conversations_deleted: int
    messages_deleted: int
    message: str


# Dependency to verify admin role
async def verify_admin(current_user: Optional[Dict] = Depends(get_current_user)):
    """Verify that the current user has admin role"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return current_user


def ensure_backup_dir():
    """Ensure backup directory exists"""
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_backup(content: str) -> str:
    """Create a backup of the current system prompt"""
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"system_prompt_{timestamp}.txt"
    filepath = os.path.join(BACKUP_DIR, filename)

    with open(filepath, 'w') as f:
        f.write(content)

    return filename


@router.get("/system-prompt", response_model=SystemPromptResponse)
async def get_system_prompt():
    """Get the current system prompt (raw template with {context} placeholder)"""
    raw_prompts = prompt_manager.get_raw_prompts()
    content = raw_prompts.get("system", "")

    # Get last modified time from prompts file
    last_modified = None
    if prompt_manager.prompts_file.exists():
        mtime = os.path.getmtime(prompt_manager.prompts_file)
        last_modified = datetime.fromtimestamp(mtime).isoformat()

    return SystemPromptResponse(content=content, last_modified=last_modified)


@router.put("/system-prompt")
async def update_system_prompt(request: SystemPromptRequest):
    """Update the system prompt (creates backup of previous)"""
    # Get current prompt and create backup if it exists
    raw_prompts = prompt_manager.get_raw_prompts()
    current_prompt = raw_prompts.get("system", "")
    if current_prompt:
        backup_filename = create_backup(current_prompt)
    else:
        backup_filename = None

    # Update the prompt using prompt manager
    prompt_manager.update_system_prompt(request.content)

    return {
        "success": True,
        "backup_created": backup_filename,
        "message": "System prompt updated successfully"
    }


@router.get("/system-prompt/backups", response_model=List[BackupInfo])
async def list_backups():
    """List all system prompt backups"""
    ensure_backup_dir()
    backups = []

    for filename in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if filename.startswith("system_prompt_") and filename.endswith(".txt"):
            filepath = os.path.join(BACKUP_DIR, filename)
            stat = os.stat(filepath)

            # Parse timestamp from filename
            try:
                timestamp_str = filename.replace("system_prompt_", "").replace(".txt", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                backups.append(BackupInfo(
                    filename=filename,
                    timestamp=timestamp.isoformat(),
                    size=stat.st_size
                ))
            except:
                continue

    return backups


@router.get("/system-prompt/backup/{filename}")
async def get_backup(filename: str):
    """Get content of a specific backup"""
    filepath = os.path.join(BACKUP_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Backup not found")

    # Security check - ensure filename is valid
    if not filename.startswith("system_prompt_") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid backup filename")

    with open(filepath, 'r') as f:
        content = f.read()

    return {"filename": filename, "content": content}


@router.post("/system-prompt/restore/{filename}")
async def restore_backup(filename: str):
    """Restore system prompt from a backup"""
    filepath = os.path.join(BACKUP_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Backup not found")

    # Security check
    if not filename.startswith("system_prompt_") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid backup filename")

    # Create backup of current before restoring
    raw_prompts = prompt_manager.get_raw_prompts()
    current_prompt = raw_prompts.get("system", "")
    if current_prompt:
        create_backup(current_prompt)

    # Read and restore backup
    with open(filepath, 'r') as f:
        content = f.read()

    prompt_manager.update_system_prompt(content)

    return {"success": True, "message": f"Restored from {filename}"}


# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/users", response_model=List[UserListItem])
async def list_all_users(admin: Dict = Depends(verify_admin)):
    """List all users in the system (admin only)"""
    try:
        query = """
            SELECT id, username, email, role, created_at, last_login, is_active
            FROM users
            ORDER BY created_at DESC
        """
        rows = await db.fetch(query)

        users = []
        for row in rows:
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            users.append(UserListItem(**row_dict))

        return users

    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: Dict = Depends(verify_admin)):
    """Delete a user and all associated data (admin only)"""
    try:
        # Prevent deleting the admin user
        user = await UserProfileManager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.role == "admin":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete admin users. Please demote the user first."
            )

        # Delete user (cascading deletes will handle related data)
        deleted = await UserProfileManager.delete_user(user_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Admin {admin.get('username')} deleted user {user.username} ({user_id})")

        return {
            "success": True,
            "message": f"User {user.username} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


# ============================================================================
# NEGOTIATION MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/negotiations", response_model=List[NegotiationListItem])
async def list_all_negotiations(admin: Dict = Depends(verify_admin)):
    """List all negotiations across all users (admin only)"""
    try:
        query = """
            SELECT n.id, n.user_id, u.username, n.title, n.status, n.created_at, n.updated_at
            FROM negotiations n
            JOIN users u ON n.user_id = u.id
            ORDER BY n.created_at DESC
        """
        rows = await db.fetch(query)

        negotiations = []
        for row in rows:
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            row_dict['user_id'] = str(row_dict['user_id'])
            negotiations.append(NegotiationListItem(**row_dict))

        return negotiations

    except Exception as e:
        logger.error(f"Failed to list negotiations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve negotiations")


@router.get("/negotiations/user/{user_id}", response_model=List[NegotiationListItem])
async def list_user_negotiations(user_id: str, admin: Dict = Depends(verify_admin)):
    """List all negotiations for a specific user (admin only)"""
    try:
        query = """
            SELECT n.id, n.user_id, u.username, n.title, n.status, n.created_at, n.updated_at
            FROM negotiations n
            JOIN users u ON n.user_id = u.id
            WHERE n.user_id = $1
            ORDER BY n.created_at DESC
        """
        rows = await db.fetch(query, user_id)

        negotiations = []
        for row in rows:
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            row_dict['user_id'] = str(row_dict['user_id'])
            negotiations.append(NegotiationListItem(**row_dict))

        return negotiations

    except Exception as e:
        logger.error(f"Failed to list user negotiations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve negotiations")


@router.delete("/negotiations/{negotiation_id}")
async def delete_negotiation(negotiation_id: str, admin: Dict = Depends(verify_admin)):
    """Delete a negotiation and all associated conversations (admin only)"""
    try:
        # Check if negotiation exists
        negotiation = await db.fetchrow(
            "SELECT id, title FROM negotiations WHERE id = $1",
            negotiation_id
        )

        if not negotiation:
            raise HTTPException(status_code=404, detail="Negotiation not found")

        # Delete negotiation (cascading deletes will handle conversations and messages)
        await db.execute("DELETE FROM negotiations WHERE id = $1", negotiation_id)

        logger.info(f"Admin {admin.get('username')} deleted negotiation {negotiation['title']} ({negotiation_id})")

        return {
            "success": True,
            "message": f"Negotiation '{negotiation['title']}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete negotiation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete negotiation")


# ============================================================================
# USAGE STATISTICS ENDPOINTS
# ============================================================================

@router.get("/usage/summary", response_model=List[UsageStats])
async def get_usage_summary(admin: Dict = Depends(verify_admin)):
    """Get usage statistics for all users (admin only)"""
    try:
        query = """
            SELECT
                u.id as user_id,
                u.username,
                COUNT(ul.id) as total_requests,
                COALESCE(SUM(ul.total_tokens), 0) as total_tokens,
                COALESCE(SUM(ul.cost), 0) as total_cost,
                ARRAY_AGG(DISTINCT ul.model) FILTER (WHERE ul.model IS NOT NULL) as models_used,
                MAX(ul.timestamp) as last_activity
            FROM users u
            LEFT JOIN usage_logs ul ON u.id = ul.user_id
            GROUP BY u.id, u.username
            ORDER BY total_requests DESC
        """
        rows = await db.fetch(query)

        stats = []
        for row in rows:
            row_dict = dict(row)
            row_dict['user_id'] = str(row_dict['user_id'])
            # Handle empty models_used array
            if row_dict['models_used'] is None:
                row_dict['models_used'] = []
            stats.append(UsageStats(**row_dict))

        return stats

    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage statistics")


@router.get("/usage/user/{user_id}", response_model=UsageStats)
async def get_user_usage(user_id: str, admin: Dict = Depends(verify_admin)):
    """Get detailed usage statistics for a specific user (admin only)"""
    try:
        query = """
            SELECT
                u.id as user_id,
                u.username,
                COUNT(ul.id) as total_requests,
                COALESCE(SUM(ul.total_tokens), 0) as total_tokens,
                COALESCE(SUM(ul.cost), 0) as total_cost,
                ARRAY_AGG(DISTINCT ul.model) FILTER (WHERE ul.model IS NOT NULL) as models_used,
                MAX(ul.timestamp) as last_activity
            FROM users u
            LEFT JOIN usage_logs ul ON u.id = ul.user_id
            WHERE u.id = $1
            GROUP BY u.id, u.username
        """
        row = await db.fetchrow(query, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        row_dict = dict(row)
        row_dict['user_id'] = str(row_dict['user_id'])
        if row_dict['models_used'] is None:
            row_dict['models_used'] = []

        return UsageStats(**row_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user statistics")


# ============================================================================
# DATABASE MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/database/reset", response_model=DatabaseResetResponse)
async def reset_database(admin: Dict = Depends(verify_admin)):
    """
    Reset the entire database EXCEPT admin users (admin only).

    WARNING: This is destructive and cannot be undone!
    Deletes all non-admin users, negotiations, conversations, and messages.
    """
    try:
        # Count records before deletion
        users_count = await db.fetchval("SELECT COUNT(*) FROM users WHERE role != 'admin'")
        negotiations_count = await db.fetchval("SELECT COUNT(*) FROM negotiations")
        conversations_count = await db.fetchval("SELECT COUNT(*) FROM conversations")
        messages_count = await db.fetchval("SELECT COUNT(*) FROM chat_messages")

        # Delete all non-admin users (cascading deletes will handle related data)
        await db.execute("DELETE FROM users WHERE role != 'admin'")

        # Delete any orphaned data (in case of cascading issues)
        await db.execute("DELETE FROM negotiations WHERE user_id NOT IN (SELECT id FROM users)")
        await db.execute("DELETE FROM sessions WHERE user_id NOT IN (SELECT id FROM users)")

        logger.warning(
            f"Admin {admin.get('username')} reset the database. "
            f"Deleted: {users_count} users, {negotiations_count} negotiations, "
            f"{conversations_count} conversations, {messages_count} messages"
        )

        return DatabaseResetResponse(
            success=True,
            users_deleted=users_count,
            negotiations_deleted=negotiations_count,
            conversations_deleted=conversations_count,
            messages_deleted=messages_count,
            message="Database reset successfully. All non-admin data has been deleted."
        )

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")
