"""
Audio Spoofing Detection Module for DF-Guard

This module implements audio spoofing detection using spectral analysis
and machine learning techniques.
"""

import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioSpoofingDetector:
    """
    Audio spoofing detection system that analyzes audio files
    and provides spoofing likelihood scores.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Model needs to be trained.")
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract audio features for spoofing detection.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Feature vector
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract various audio features
            features = []
            
            # 1. MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # 4. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend([
                np.mean(contrast, axis=1),
                np.std(contrast, axis=1)
            ])
            
            # 5. Tonnetz (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features.extend([
                np.mean(tonnetz, axis=1),
                np.std(tonnetz, axis=1)
            ])
            
            # Flatten all features
            feature_vector = np.concatenate([np.array(f).flatten() for f in features])
            
            return feature_vector
        
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            # Return zero vector if feature extraction fails
            return np.zeros(200)  # Approximate feature dimension
    
    def train_model(self, audio_files: List[str], labels: List[int]):
        """
        Train the audio spoofing detection model.
        
        Args:
            audio_files: List of audio file paths
            labels: List of labels (0 for genuine, 1 for spoof)
        """
        logger.info(f"Training model with {len(audio_files)} samples")
        
        # Extract features from all audio files
        features = []
        valid_labels = []
        
        for audio_file, label in zip(audio_files, labels):
            feature_vector = self.extract_features(audio_file)
            if feature_vector is not None and len(feature_vector) > 0:
                features.append(feature_vector)
                valid_labels.append(label)
        
        if not features:
            logger.error("No valid features extracted. Cannot train model.")
            return
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(valid_labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("Model training completed")
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict if an audio file is spoofed.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            # Use heuristic-based detection if model is not trained
            return self._heuristic_detection(audio_path)
        
        # Extract features
        features = self.extract_features(audio_path)
        if features is None or len(features) == 0:
            return {
                'spoofing_score': 0.5,
                'confidence': 0.0,
                'error': 'Feature extraction failed'
            }
        
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction
        probabilities = self.model.predict_proba(features_scaled)[0]
        spoofing_score = probabilities[1]  # Probability of being spoof
        
        # Calculate confidence based on how far the score is from 0.5
        confidence = abs(spoofing_score - 0.5) * 2
        
        return {
            'spoofing_score': spoofing_score,
            'confidence': confidence,
            'prediction': 'spoof' if spoofing_score > 0.5 else 'genuine'
        }
    
    def _heuristic_detection(self, audio_path: str) -> Dict:
        """
        Heuristic-based spoofing detection when no trained model is available.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with detection results
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate various heuristic measures
            
            # 1. Spectral flatness (measure of how noise-like vs. tone-like)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # 2. Zero crossing rate variability
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_var = np.var(zcr)
            
            # 3. Spectral rolloff variability
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_var = np.var(rolloff)
            
            # 4. Energy distribution
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            energy_var = np.var(np.sum(magnitude, axis=0))
            
            # Simple heuristic scoring
            # Higher spectral flatness might indicate synthetic audio
            # Higher variability in features might indicate natural speech
            
            flatness_score = min(1.0, spectral_flatness * 10)  # Normalize
            variability_score = 1.0 - min(1.0, (zcr_var + rolloff_var + energy_var) / 3)
            
            # Combine scores (this is a simplified heuristic)
            spoofing_score = (flatness_score + variability_score) / 2
            
            # Add some randomness to simulate model uncertainty
            spoofing_score += np.random.normal(0, 0.1)
            spoofing_score = np.clip(spoofing_score, 0, 1)
            
            confidence = 0.6  # Lower confidence for heuristic method
            
            return {
                'spoofing_score': spoofing_score,
                'confidence': confidence,
                'prediction': 'spoof' if spoofing_score > 0.5 else 'genuine',
                'method': 'heuristic',
                'features': {
                    'spectral_flatness': spectral_flatness,
                    'zcr_variance': zcr_var,
                    'rolloff_variance': rolloff_var,
                    'energy_variance': energy_var
                }
            }
        
        except Exception as e:
            logger.error(f"Error in heuristic detection for {audio_path}: {e}")
            return {
                'spoofing_score': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def save_model(self, model_path: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            logger.warning("Model is not trained. Nothing to save.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")

def create_sample_training_data():
    """Create sample training data for demonstration."""
    # This would normally use real datasets like ASVspoof
    # For demo purposes, we'll use the sample audio files we created
    
    sample_dir = Path('data/media/audio_clips')
    if not sample_dir.exists():
        logger.error("Sample audio directory not found")
        return [], []
    
    audio_files = []
    labels = []
    
    # Assign labels to sample files (alternating genuine/spoof for demo)
    for i, audio_file in enumerate(sample_dir.glob('*.wav')):
        audio_files.append(str(audio_file))
        labels.append(i % 2)  # Alternate between 0 (genuine) and 1 (spoof)
    
    return audio_files, labels

def main():
    """Test the audio spoofing detector."""
    detector = AudioSpoofingDetector()
    
    # Test with sample audio files
    sample_dir = Path('data/media/audio_clips')
    if sample_dir.exists():
        print("Testing audio spoofing detection:")
        
        for audio_file in sample_dir.glob('*.wav'):
            result = detector.predict(str(audio_file))
            print(f"\nAudio: {audio_file.name}")
            print(f"Spoofing Score: {result['spoofing_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Prediction: {result['prediction']}")
            if 'method' in result:
                print(f"Method: {result['method']}")
        
        # Optionally train a model with sample data
        print("\n" + "="*50)
        print("Training model with sample data...")
        
        audio_files, labels = create_sample_training_data()
        if audio_files:
            detector.train_model(audio_files, labels)
            
            # Test again with trained model
            print("\nTesting with trained model:")
            for audio_file in sample_dir.glob('*.wav'):
                result = detector.predict(str(audio_file))
                print(f"\nAudio: {audio_file.name}")
                print(f"Spoofing Score: {result['spoofing_score']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Prediction: {result['prediction']}")
    
    else:
        print("No sample audio files found. Please run the dataset preparation script first.")

if __name__ == '__main__':
    main()

