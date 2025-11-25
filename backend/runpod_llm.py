"""
RunPod LLM wrapper for LangChain compatibility.

This provides a basic fallback LLM when no other backends (OpenAI, Anthropic, Ollama) are available.
Uses RunPod's serverless inference API.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class ChatRunPod(LLM):
    """
    LangChain-compatible LLM wrapper for RunPod's serverless inference API.

    This is designed as a basic fallback model when no other LLM backends are configured.

    Environment Variables:
        RUNPOD_API_KEY: API key for RunPod (required)
        RUNPOD_ENDPOINT_ID: The endpoint ID for the deployed model (required)

    Example usage:
        llm = ChatRunPod()
        response = llm.invoke("What is negotiation?")
    """

    api_key: Optional[str] = None
    endpoint_id: Optional[str] = None
    base_url: str = "https://api.runpod.ai/v2"
    timeout: int = 120

    # Default fallback values (can be overridden via environment)
    _default_endpoint_id: str = "2m77lcg44ccrb6"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("RUNPOD_API_KEY")
        if not self.endpoint_id:
            self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", self._default_endpoint_id)

        if not self.api_key:
            raise ValueError(
                "RunPod API key not found. Set RUNPOD_API_KEY environment variable."
            )

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "runpod"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
        }

    def _format_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert LangChain messages to a single prompt string.

        This handles the conversion from chat format to a simple prompt
        that RunPod's basic API expects.
        """
        prompt_parts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}\n")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"User: {message.content}\n")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}\n")
            else:
                # Generic handling for other message types
                prompt_parts.append(f"{message.content}\n")

        # Add prompt for assistant response
        prompt_parts.append("Assistant:")

        return "".join(prompt_parts)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a synchronous call to RunPod's API.

        Args:
            prompt: The input prompt
            stop: Optional list of stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments

        Returns:
            The generated text response
        """
        url = f"{self.base_url}/{self.endpoint_id}/runsync"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "input": {
                "prompt": prompt,
            }
        }

        # Add stop sequences if provided
        if stop:
            payload["input"]["stop"] = stop

        logger.debug(f"Calling RunPod endpoint: {url}")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            if "output" in result:
                output = result["output"]
                # Output could be a string or a dict with 'text' key
                if isinstance(output, str):
                    return output
                elif isinstance(output, dict):
                    return output.get("text", output.get("response", str(output)))
                elif isinstance(output, list) and len(output) > 0:
                    return str(output[0])
                else:
                    return str(output)
            elif "error" in result:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"RunPod API error: {error_msg}")
                raise RuntimeError(f"RunPod API error: {error_msg}")
            else:
                logger.warning(f"Unexpected RunPod response format: {result}")
                return str(result)

        except requests.exceptions.Timeout:
            logger.error(f"RunPod request timed out after {self.timeout}s")
            raise RuntimeError(f"RunPod request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"RunPod request failed: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")

    def invoke(self, input: Any, **kwargs) -> str:
        """
        Invoke the LLM with input.

        Handles both string prompts and LangChain message lists.
        """
        if isinstance(input, str):
            return self._call(input, **kwargs)
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            prompt = self._format_messages_to_prompt(input)
            return self._call(prompt, **kwargs)
        elif hasattr(input, "to_messages"):
            # Handle prompt templates
            messages = input.to_messages()
            prompt = self._format_messages_to_prompt(messages)
            return self._call(prompt, **kwargs)
        else:
            # Try to convert to string
            return self._call(str(input), **kwargs)


def is_runpod_available() -> bool:
    """
    Check if RunPod is available as a fallback.

    Returns True if RUNPOD_API_KEY is set.
    """
    return bool(os.getenv("RUNPOD_API_KEY"))


def get_runpod_llm(**kwargs) -> ChatRunPod:
    """
    Factory function to create a RunPod LLM instance.

    Args:
        **kwargs: Optional overrides for api_key, endpoint_id, etc.

    Returns:
        Configured ChatRunPod instance
    """
    return ChatRunPod(**kwargs)
