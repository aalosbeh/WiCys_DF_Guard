"""
Evidence Card Module for DF-Guard

This module defines the EvidenceCard class, which is a standardized
data structure for storing and organizing the results of the multimodal
analysis.
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import uuid
import numpy as np

class EvidenceCard:
    """
    A data structure for storing analysis results in a structured and
    explainable format.

    Each card includes a unique ID, timestamp, overall risk assessment,
    modality-specific results, suspicious elements, and actionable
    recommendations.
    """

    def __init__(self, model_versions: Dict[str, str] = None, analysis_params: Dict[str, Any] = None):
        """
        Initializes the EvidenceCard.

        Args:
            model_versions: A dictionary of model names and their versions.
            analysis_params: A dictionary of parameters used during the analysis.
        """
        self.card_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.overall_risk_score = 0.0
        self.confidence = 0.0
        self.risk_level = "LOW"
        self.modalities = {}
        self.suspicious_elements = []
        self.recommendations = []
        self.model_versions = model_versions or {}
        self.analysis_params = analysis_params or {}

    def add_modality_result(self, modality: str, result: Dict):
        """Adds the analysis result from a specific modality."""
        self.modalities[modality] = self._convert_numpy_types(result)

    def add_suspicious_element(self, element_type: str, score: float, description: str, details: Dict = None):
        """Adds a detected suspicious element to the card."""
        self.suspicious_elements.append({
            'type': element_type,
            'score': float(score),
            'description': description,
            'details': self._convert_numpy_types(details or {})
        })

    def add_recommendation(self, recommendation: str, priority: str = "MEDIUM"):
        """Adds an actionable recommendation for security analysts."""
        self.recommendations.append({
            'recommendation': recommendation,
            'priority': priority
        })

    def _convert_numpy_types(self, data: Any) -> Any:
        """Recursively converts numpy types to native Python types."""
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(i) for i in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def to_dict(self) -> Dict:
        """Converts the EvidenceCard to a dictionary."""
        return self._convert_numpy_types({
            'card_id': self.card_id,
            'timestamp': self.timestamp,
            'overall_risk_score': self.overall_risk_score,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'modalities': self.modalities,
            'suspicious_elements': self.suspicious_elements,
            'recommendations': self.recommendations,
            'metadata': {
                'model_versions': self.model_versions,
                'analysis_params': self.analysis_params
            }
        })

    def to_json(self, indent: int = 2) -> str:
        """Converts the EvidenceCard to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
