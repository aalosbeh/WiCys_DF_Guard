"""
Input Validation Module for DF-Guard

This module provides functions for validating the inputs to the DF-Guard
analysis pipeline. This helps ensure that the pipeline receives
well-formed data and can handle potential errors gracefully.
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac']

def validate_file_path(path: Optional[str], supported_extensions: List[str]) -> bool:
    """
    Validates that a file path exists, is a file, and has a supported extension.

    Args:
        path: The file path to validate.
        supported_extensions: A list of supported file extensions.

    Returns:
        True if the path is a valid file, False otherwise.
    """
    if not path:
        return False

    file = Path(path)

    if not file.is_file():
        logger.warning(f"File not found or is not a regular file: {path}")
        return False

    if file.suffix.lower() not in supported_extensions:
        logger.warning(f"Unsupported file extension: {file.suffix}. Supported extensions are: {supported_extensions}")
        return False

    return True

def validate_text_content(text: Optional[str]) -> bool:
    """
    Validates that the text content is not empty or just whitespace.

    Args:
        text: The text content to validate.

    Returns:
        True if the text is valid, False otherwise.
    """
    if not text or not text.strip():
        logger.warning("Text content is empty or contains only whitespace.")
        return False

    return True

def validate_fusion_weights(weights: Dict[str, float]) -> bool:
    """
    Validates the fusion weights to ensure they are properly configured.

    Args:
        weights: A dictionary of modality weights.

    Returns:
        True if the weights are valid, False otherwise.
    """
    if not isinstance(weights, dict):
        logger.error("Fusion weights must be a dictionary.")
        return False

    total_weight = sum(weights.values())

    if total_weight <= 0:
        logger.error("The sum of fusion weights must be positive.")
        return False

    if not all(w >= 0 for w in weights.values()):
        logger.error("All fusion weights must be non-negative.")
        return False

    return True
