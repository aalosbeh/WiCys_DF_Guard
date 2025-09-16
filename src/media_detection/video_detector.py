"""
Video Deepfake Detection Module for DF-Guard

This module implements video deepfake detection using pre-trained models
and frame-level analysis techniques.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoDetector(nn.Module):
    """
    Simple CNN-based video deepfake detector.
    This is a lightweight implementation for demonstration purposes.
    """
    
    def __init__(self, num_classes=2):
        super(SimpleVideoDetector, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VideoDeepfakeDetector:
    """
    Video deepfake detection system that analyzes video frames
    and provides deepfake likelihood scores.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = SimpleVideoDetector()
        self.model.to(self.device)
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No pre-trained model found. Using random initialization.")
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_face_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract face regions from a video frame using OpenCV's face detector.
        
        Args:
            frame: Input video frame (BGR format)
        
        Returns:
            List of face regions
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_regions = []
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                face_regions.append(face_region)
        
        return face_regions
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single video frame for deepfake artifacts.
        
        Args:
            frame: Input video frame (BGR format)
        
        Returns:
            Dictionary with analysis results
        """
        # Extract face regions
        face_regions = self.extract_face_regions(frame)
        
        if not face_regions:
            return {
                'deepfake_score': 0.5,  # Neutral score when no faces detected
                'confidence': 0.0,
                'num_faces': 0,
                'artifacts': []
            }
        
        # Analyze each face region
        face_scores = []
        for face_region in face_regions:
            # Preprocess face region
            face_tensor = self.transform(face_region).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                deepfake_prob = probabilities[0][1].item()  # Probability of being fake
                face_scores.append(deepfake_prob)
        
        # Aggregate scores from all faces
        avg_score = np.mean(face_scores)
        max_score = np.max(face_scores)
        confidence = 1.0 - np.std(face_scores) if len(face_scores) > 1 else 0.8
        
        # Detect potential artifacts (simplified)
        artifacts = self._detect_artifacts(frame, face_regions)
        
        return {
            'deepfake_score': max_score,  # Use maximum score as overall score
            'confidence': confidence,
            'num_faces': len(face_regions),
            'face_scores': face_scores,
            'artifacts': artifacts
        }
    
    def _detect_artifacts(self, frame: np.ndarray, face_regions: List[np.ndarray]) -> List[str]:
        """
        Detect potential deepfake artifacts in the frame.
        
        Args:
            frame: Input video frame
            face_regions: List of detected face regions
        
        Returns:
            List of detected artifacts
        """
        artifacts = []
        
        # Check for blurriness (common in deepfakes)
        for i, face_region in enumerate(face_regions):
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            if laplacian_var < 100:  # Threshold for blurriness
                artifacts.append(f'face_{i}_blurry')
        
        # Check for color inconsistencies (simplified)
        if len(face_regions) > 0:
            frame_mean_color = np.mean(frame, axis=(0, 1))
            for i, face_region in enumerate(face_regions):
                face_mean_color = np.mean(face_region, axis=(0, 1))
                color_diff = np.linalg.norm(frame_mean_color - face_mean_color)
                
                if color_diff > 50:  # Threshold for color inconsistency
                    artifacts.append(f'face_{i}_color_inconsistent')
        
        return artifacts
    
    def analyze_video(self, video_path: str, sample_rate: float = 0.1, max_frames: int = 100) -> Dict:
        """
        Analyze an entire video for deepfake content.
        
        Args:
            video_path: Path to video file
            sample_rate: Fraction of frames to analyze
            max_frames: Maximum number of frames to analyze
        
        Returns:
            Dictionary with video analysis results
        """
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {
                'deepfake_score': 0.5,
                'confidence': 0.0,
                'error': 'Could not open video'
            }
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate which frames to analyze
        frames_to_analyze = min(int(total_frames * sample_rate), max_frames)
        frame_indices = np.linspace(0, total_frames - 1, frames_to_analyze, dtype=int)
        
        # Analyze selected frames
        frame_results = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                result = self.analyze_frame(frame)
                result['frame_idx'] = frame_idx
                result['timestamp'] = frame_idx / fps
                frame_results.append(result)
        
        cap.release()
        
        if not frame_results:
            return {
                'deepfake_score': 0.5,
                'confidence': 0.0,
                'error': 'No frames could be analyzed'
            }
        
        # Aggregate results across all frames
        frame_scores = [r['deepfake_score'] for r in frame_results]
        frame_confidences = [r['confidence'] for r in frame_results]
        
        # Temporal consistency analysis
        score_variance = np.var(frame_scores)
        temporal_consistency = 1.0 / (1.0 + score_variance)  # Higher variance = lower consistency
        
        # Overall video score (weighted average with temporal consistency)
        avg_score = np.mean(frame_scores)
        max_score = np.max(frame_scores)
        final_score = 0.7 * avg_score + 0.3 * max_score
        
        # Overall confidence
        avg_confidence = np.mean(frame_confidences)
        final_confidence = avg_confidence * temporal_consistency
        
        # Collect all artifacts
        all_artifacts = []
        for result in frame_results:
            all_artifacts.extend(result['artifacts'])
        
        return {
            'video_path': video_path,
            'deepfake_score': final_score,
            'confidence': final_confidence,
            'temporal_consistency': temporal_consistency,
            'frames_analyzed': len(frame_results),
            'avg_faces_per_frame': np.mean([r['num_faces'] for r in frame_results]),
            'artifacts': list(set(all_artifacts)),  # Remove duplicates
            'frame_results': frame_results
        }

def main():
    """Test the video deepfake detector."""
    detector = VideoDeepfakeDetector()
    
    # Test with sample videos if they exist
    sample_dir = Path('data/media/frames')
    if sample_dir.exists():
        for video_file in sample_dir.glob('*.mp4'):
            result = detector.analyze_video(str(video_file))
            print(f"\nVideo: {video_file.name}")
            print(f"Deepfake Score: {result['deepfake_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Frames Analyzed: {result['frames_analyzed']}")
            print(f"Artifacts: {result['artifacts']}")
    else:
        print("No sample videos found. Please run the dataset preparation script first.")

if __name__ == '__main__':
    main()

