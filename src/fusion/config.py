"""
Configuration settings for the DF-Guard multimodal fusion system.

This file centralizes key parameters, such as fusion weights and risk thresholds,
to allow for easier tuning and maintenance.
"""

# Fusion weights for combining modality scores
# These weights can be tuned based on validation data to optimize performance.
# The sum of these weights should ideally be 1.0, though the fusion logic
# normalizes them to prevent errors if they don't.
FUSION_WEIGHTS = {
    'video': 0.4,
    'audio': 0.3,
    'text': 0.3
}

# Risk thresholds for classifying the final risk score into levels
# These thresholds determine the boundaries for LOW, MEDIUM, and HIGH risk levels.
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Settings for suspicious element detection
# These thresholds determine when a modality's result is considered suspicious
# enough to be added to the evidence card.
SUSPICIOUS_THRESHOLDS = {
    'video_deepfake': 0.6,
    'audio_spoofing': 0.6,
    'text_injection': 0.5
}
