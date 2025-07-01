import json
import os
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages system and user prompts stored in JSON configuration files.
    Provides clean separation between system instructions and user messages.
    """
    
    def __init__(self, config_dir: str = ".config"):
        self.config_dir = Path(config_dir)
        self.prompts_file = self.config_dir / "prompts.json"
        self._prompts = None
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load prompts on initialization
        self.load_prompts()
    
    def load_prompts(self) -> Dict[str, str]:
        """Load prompts from JSON configuration file"""
        try:
            if self.prompts_file.exists():
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    self._prompts = json.load(f)
                logger.info(f"Loaded prompts from {self.prompts_file}")
            else:
                # Create default prompts if file doesn't exist
                self._prompts = self.get_default_prompts()
                self.save_prompts()
                logger.info(f"Created default prompts at {self.prompts_file}")
            
            return self._prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            # Fallback to default prompts
            self._prompts = self.get_default_prompts()
            return self._prompts
    
    def save_prompts(self):
        """Save current prompts to JSON configuration file"""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self._prompts, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved prompts to {self.prompts_file}")
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
    
    def get_default_prompts(self) -> Dict[str, str]:
        """Get default system and user prompts"""
        return {
            "system": """You are a skilled sales negotiator and expert advisor, leveraging principles from mainstream negotiation books like 'Getting Past No', 'The Upward Spiral', 'Getting to Yes', 'Never Split the Difference', and 'How to Win Friends and Influence People.' You carefully analyze client communications to craft empathetic yet assertive responses.

Your goal is to find win-win solutions while ensuring the best outcomes for the user. You prioritize understanding the client's needs and concerns, and help apply negotiation strategies like building rapport, uncovering underlying interests, and leveraging tactical empathy. You must avoid being overly aggressive or dismissive of client positions, always focusing on maintaining positive relationships while negotiating favorable terms.

You have access to expert negotiation knowledge from the following sources:
{context}

Each response MUST include:
• A detailed breakdown of the negotiation so far and piece-by-piece analysis of the client's communication
• A fully composed draft response to the client when appropriate
• A bullet list of calibrated questions to use in the negotiation
• Potential client responses and suggested actions for each scenario
• PLEASE Framework self-assessment (Polite, Logical, Empathetic, Assertive, Strategic, Engaging - score each /5)

Your responses must be POLITE, LOGICAL, EMPATHETIC, ASSERTIVE, STRATEGIC, and ENGAGING. The tone should remain professional but non-formal to foster ease and approachability.

Provide comprehensive negotiation guidance based on the user's specific situation.""",
            
            "user": """Please analyze this negotiation situation and provide expert guidance:

{question}

Based on your expertise and the negotiation principles in your knowledge base, please provide detailed advice following the structure outlined in your instructions."""
        }
    
    def get_system_prompt(self, context: str = "") -> str:
        """Get the system prompt with context filled in"""
        if not self._prompts:
            self.load_prompts()
        
        system_prompt = self._prompts.get("system", "")
        return system_prompt.format(context=context)
    
    def get_user_prompt(self, question: str) -> str:
        """Get the user prompt with question filled in"""
        if not self._prompts:
            self.load_prompts()
        
        user_prompt = self._prompts.get("user", "")
        return user_prompt.format(question=question)
    
    def get_prompts_for_chat(self, question: str, context: str = "") -> Tuple[str, str]:
        """
        Get both system and user prompts formatted for chat completion.
        Returns (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt(context=context)
        user_prompt = self.get_user_prompt(question=question)
        return system_prompt, user_prompt
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt"""
        if not self._prompts:
            self.load_prompts()
        
        self._prompts["system"] = new_prompt
        self.save_prompts()
        logger.info("Updated system prompt")
    
    def update_user_prompt(self, new_prompt: str):
        """Update the user prompt template"""
        if not self._prompts:
            self.load_prompts()
        
        self._prompts["user"] = new_prompt
        self.save_prompts()
        logger.info("Updated user prompt")
    
    def get_raw_prompts(self) -> Dict[str, str]:
        """Get the raw prompt templates (with placeholders)"""
        if not self._prompts:
            self.load_prompts()
        return self._prompts.copy()
    
    def validate_prompts(self) -> Dict[str, bool]:
        """Validate that prompts have required placeholders"""
        if not self._prompts:
            self.load_prompts()
        
        validation = {
            "system_has_context": "{context}" in self._prompts.get("system", ""),
            "user_has_question": "{question}" in self._prompts.get("user", ""),
            "both_prompts_exist": bool(self._prompts.get("system")) and bool(self._prompts.get("user"))
        }
        
        return validation
    
    def get_prompt_info(self) -> Dict[str, any]:
        """Get information about current prompts for debugging/admin"""
        if not self._prompts:
            self.load_prompts()
        
        validation = self.validate_prompts()
        
        return {
            "prompts_file": str(self.prompts_file),
            "prompts_exist": self.prompts_file.exists(),
            "system_prompt_length": len(self._prompts.get("system", "")),
            "user_prompt_length": len(self._prompts.get("user", "")),
            "validation": validation,
            "last_modified": self.prompts_file.stat().st_mtime if self.prompts_file.exists() else None
        }