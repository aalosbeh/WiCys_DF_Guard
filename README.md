# DF-Guard: Multimodal Guardrails for Deepfake-Enhanced Social Engineering

## Overview

DF-Guard is a multimodal security system that combines deepfake detection (video/audio) with prompt injection detection (text) to identify sophisticated social engineering attacks. Modern scammers increasingly use AI-generated media combined with carefully crafted text to conduct phishing, CEO fraud, and romance scams. This research develops a practical defense system that security teams can actually deploy.

## Features

- **Video Deepfake Detection**: Detects manipulated video content using state-of-the-art deep learning models
- **Audio Spoofing Detection**: Identifies synthetic or manipulated audio content
- **Text Injection Detection**: Detects prompt injection attempts in text content
- **Multimodal Fusion**: Combines evidence from all modalities for improved accuracy
- **Explainable AI**: Provides evidence cards with highlighted suspicious content
- **Real-time Processing**: Designed for deployment in security operations centers

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/df-guard.git
cd df-guard

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py
```

## Quick Start

```python
from src.fusion.risk_scorer import DFGuardPipeline

# Initialize the pipeline
pipeline = DFGuardPipeline()

# Analyze a sample
result = pipeline.analyze(
    video_path="sample_video.mp4",
    audio_path="sample_audio.wav",
    text_content="Sample text content"
)

print(f"Risk Score: {result.risk_score}")
print(f"Evidence: {result.evidence_card}")
```

## Dataset

The project uses several public datasets:
- FaceForensics++ for video deepfake detection
- ASVspoof for audio spoofing detection
- Custom prompt injection dataset

## Citation

If you use this work in your research, please cite:

```bibtex
@article{dfguard2024,
  title={DF-Guard: Multimodal Guardrails for Deepfake-Enhanced Social Engineering},
  author={WiCyS Research Team},
  journal={WiCyS Annual Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

