"""
Admin API Routes

Provides admin-only endpoints for system configuration.
Only accessible to users with role='admin'.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import json

from backend.prompt_manager import PromptManager

router = APIRouter(prefix="/admin", tags=["admin"])

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
