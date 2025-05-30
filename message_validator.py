"""
Message validator for LiveKit OpenAI integration.

This module provides functions to validate and fix messages before they are sent to the OpenAI API.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("message_validator")

def validate_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix a message to ensure it has a valid 'content' key.
    
    Args:
        message: The message to validate
        
    Returns:
        The validated message
    """
    # If message doesn't have a content key or content is None, add an empty string
    if 'content' not in message or message['content'] is None:
        message['content'] = ""
        logger.debug(f"Added empty content to message with role: {message.get('role', 'unknown')}")
    
    # Ensure content is a string
    if not isinstance(message['content'], str):
        message['content'] = str(message['content'])
        logger.debug(f"Converted non-string content to string for message with role: {message.get('role', 'unknown')}")
    
    return message

def validate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and fix a list of messages to ensure each has a valid 'content' key.
    
    Args:
        messages: The list of messages to validate
        
    Returns:
        The validated list of messages
    """
    return [validate_message(message) for message in messages]
