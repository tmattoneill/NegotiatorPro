import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class AdminConfig:
    """Admin configuration and session management"""
    
    def __init__(self, config_file="admin_config.json"):
        self.config_file = config_file
        self.sessions_file = "admin_sessions.json"
        self.usage_file = "usage_stats.json"
        self.load_config()
        
    def load_config(self):
        """Load admin configuration"""
        default_config = {
            "admin_password_hash": self._hash_password("admin123"),  # Default password
            "system_prompt": "",
            "default_user_prompt": "",
            "session_duration_hours": 24,
            "max_upload_size_mb": 50
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # Ensure all keys exist
                for key, value in default_config.items():
                    if key not in self.config:
                        self.config[key] = value
            except Exception as e:
                print(f"Error loading admin config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save admin configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving admin config: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        """Verify admin password"""
        return self._hash_password(password) == self.config["admin_password_hash"]
    
    def change_password(self, new_password: str):
        """Change admin password"""
        self.config["admin_password_hash"] = self._hash_password(new_password)
        self.save_config()
    
    def create_session(self, session_id: str) -> str:
        """Create admin session"""
        sessions = self.load_sessions()
        expiry = datetime.now() + timedelta(hours=self.config["session_duration_hours"])
        sessions[session_id] = expiry.isoformat()
        self.save_sessions(sessions)
        return session_id
    
    def is_valid_session(self, session_id: str) -> bool:
        """Check if session is valid"""
        sessions = self.load_sessions()
        if session_id not in sessions:
            return False
        
        expiry = datetime.fromisoformat(sessions[session_id])
        if datetime.now() > expiry:
            # Remove expired session
            del sessions[session_id]
            self.save_sessions(sessions)
            return False
        
        return True
    
    def load_sessions(self) -> Dict[str, str]:
        """Load admin sessions"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_sessions(self, sessions: Dict[str, str]):
        """Save admin sessions"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        sessions = self.load_sessions()
        current_time = datetime.now()
        valid_sessions = {}
        
        for session_id, expiry_str in sessions.items():
            expiry = datetime.fromisoformat(expiry_str)
            if current_time <= expiry:
                valid_sessions[session_id] = expiry_str
        
        self.save_sessions(valid_sessions)
    
    def get_system_prompt(self) -> str:
        """Get system prompt"""
        return self.config.get("system_prompt", "")
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt"""
        self.config["system_prompt"] = prompt
        self.save_config()
    
    def get_default_user_prompt(self) -> str:
        """Get default user prompt"""
        return self.config.get("default_user_prompt", "")
    
    def set_default_user_prompt(self, prompt: str):
        """Set default user prompt"""
        self.config["default_user_prompt"] = prompt
        self.save_config()
    
    def log_usage(self, model: str, tokens_used: int, cost: float = 0.0):
        """Log API usage"""
        usage_data = self.load_usage_stats()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in usage_data:
            usage_data[today] = {"total_tokens": 0, "total_cost": 0.0, "models": {}}
        
        if model not in usage_data[today]["models"]:
            usage_data[today]["models"][model] = {"tokens": 0, "requests": 0, "cost": 0.0}
        
        usage_data[today]["total_tokens"] += tokens_used
        usage_data[today]["total_cost"] += cost
        usage_data[today]["models"][model]["tokens"] += tokens_used
        usage_data[today]["models"][model]["requests"] += 1
        usage_data[today]["models"][model]["cost"] += cost
        
        self.save_usage_stats(usage_data)
    
    def load_usage_stats(self) -> Dict[str, Any]:
        """Load usage statistics"""
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_usage_stats(self, usage_data: Dict[str, Any]):
        """Save usage statistics"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
        except Exception as e:
            print(f"Error saving usage stats: {e}")
    
    def get_usage_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for last N days"""
        usage_data = self.load_usage_stats()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        summary = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_requests": 0,
            "models": {},
            "daily_stats": []
        }
        
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            if date in usage_data:
                day_data = usage_data[date]
                summary["total_tokens"] += day_data["total_tokens"]
                summary["total_cost"] += day_data["total_cost"]
                
                for model, stats in day_data["models"].items():
                    if model not in summary["models"]:
                        summary["models"][model] = {"tokens": 0, "requests": 0, "cost": 0.0}
                    summary["models"][model]["tokens"] += stats["tokens"]
                    summary["models"][model]["requests"] += stats["requests"]
                    summary["models"][model]["cost"] += stats["cost"]
                    summary["total_requests"] += stats["requests"]
                
                summary["daily_stats"].append({
                    "date": date,
                    "tokens": day_data["total_tokens"],
                    "cost": day_data["total_cost"],
                    "models": day_data["models"]
                })
            else:
                summary["daily_stats"].append({
                    "date": date,
                    "tokens": 0,
                    "cost": 0.0,
                    "models": {}
                })
        
        return summary