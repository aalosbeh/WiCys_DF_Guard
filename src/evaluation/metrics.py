"""
Evaluation Metrics Module for DF-Guard

This module implements various evaluation metrics for deepfake detection,
audio spoofing detection, and text injection detection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for multimodal detection systems.
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            y_scores: Prediction scores (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['auc_roc'] = 0.5  # Random performance if only one class
        
        # Equal Error Rate (EER)
        if y_scores is not None:
            metrics['eer'] = self._calculate_eer(y_true, y_scores)
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_eer(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Equal Error Rate (EER)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        
        # Find the threshold where FPR and FNR are closest
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer
    
    def evaluate_modality(self, modality: str, y_true: List, y_pred: List, 
                         y_scores: Optional[List] = None) -> Dict:
        """
        Evaluate a single modality (video, audio, or text).
        
        Args:
            modality: Name of the modality
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (optional)
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {modality} modality...")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_scores is not None:
            y_scores = np.array(y_scores)
        
        # Calculate metrics
        metrics = self.calculate_binary_metrics(y_true, y_pred, y_scores)
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Add classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        # Store results
        self.results[modality] = metrics
        
        return metrics
    
    def evaluate_fusion(self, y_true: List, y_pred: List, y_scores: List,
                       individual_results: Dict) -> Dict:
        """
        Evaluate the fusion system and compare with individual modalities.
        
        Args:
            y_true: True labels for fusion
            y_pred: Predicted labels from fusion
            y_scores: Prediction scores from fusion
            individual_results: Results from individual modalities
        
        Returns:
            Dictionary of fusion evaluation results
        """
        logger.info("Evaluating fusion system...")
        
        # Calculate fusion metrics
        fusion_metrics = self.evaluate_modality('fusion', y_true, y_pred, y_scores)
        
        # Compare with individual modalities
        comparison = {}
        for modality, results in individual_results.items():
            comparison[modality] = {
                'accuracy_diff': fusion_metrics['accuracy'] - results['accuracy'],
                'f1_diff': fusion_metrics['f1_score'] - results['f1_score'],
                'auc_diff': fusion_metrics.get('auc_roc', 0) - results.get('auc_roc', 0)
            }
        
        fusion_metrics['modality_comparison'] = comparison
        
        return fusion_metrics
    
    def plot_confusion_matrices(self, save_dir: str = 'experiments/results'):
        """Plot confusion matrices for all evaluated modalities."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        n_modalities = len(self.results)
        if n_modalities == 0:
            return
        
        fig, axes = plt.subplots(1, n_modalities, figsize=(5 * n_modalities, 4))
        if n_modalities == 1:
            axes = [axes]
        
        for i, (modality, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{modality.capitalize()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {save_dir}/confusion_matrices.png")
    
    def plot_roc_curves(self, y_true_dict: Dict, y_scores_dict: Dict, 
                       save_dir: str = 'experiments/results'):
        """
        Plot ROC curves for all modalities.
        
        Args:
            y_true_dict: Dictionary of true labels for each modality
            y_scores_dict: Dictionary of prediction scores for each modality
            save_dir: Directory to save plots
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (modality, y_scores) in enumerate(y_scores_dict.items()):
            if modality in y_true_dict:
                y_true = y_true_dict[modality]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    auc = roc_auc_score(y_true, y_scores)
                    
                    plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                            label=f'{modality.capitalize()} (AUC = {auc:.3f})')
                except ValueError:
                    logger.warning(f"Could not plot ROC curve for {modality}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multimodal Detection System')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {save_dir}/roc_curves.png")
    
    def plot_performance_comparison(self, save_dir: str = 'experiments/results'):
        """Plot performance comparison across modalities."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            return
        
        # Prepare data for plotting
        modalities = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        
        data = []
        for modality in modalities:
            for metric in metrics:
                if metric in self.results[modality]:
                    data.append({
                        'Modality': modality.capitalize(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': self.results[modality][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Score', hue='Modality')
        plt.title('Performance Comparison Across Modalities')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Modality')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison saved to {save_dir}/performance_comparison.png")
    
    def generate_results_table(self, save_dir: str = 'experiments/results') -> pd.DataFrame:
        """Generate a results table for the paper."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            return pd.DataFrame()
        
        # Create results table
        table_data = []
        for modality, results in self.results.items():
            row = {
                'Modality': modality.capitalize(),
                'Accuracy': f"{results['accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'Specificity': f"{results['specificity']:.3f}",
            }
            
            if 'auc_roc' in results:
                row['AUC-ROC'] = f"{results['auc_roc']:.3f}"
            
            if 'eer' in results:
                row['EER'] = f"{results['eer']:.3f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        df.to_csv(f'{save_dir}/results_table.csv', index=False)
        
        # Save as LaTeX table
        latex_table = df.to_latex(index=False, escape=False)
        with open(f'{save_dir}/results_table.tex', 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Results table saved to {save_dir}/")
        
        return df
    
    def save_results(self, save_dir: str = 'experiments/results'):
        """Save all evaluation results to JSON."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f'{save_dir}/evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_dir}/evaluation_results.json")

def create_sample_evaluation_data():
    """Create sample evaluation data for demonstration."""
    np.random.seed(42)
    
    # Simulate evaluation results for different modalities
    n_samples = 200
    
    # Video modality (deepfake detection)
    video_true = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    video_scores = np.random.beta(2, 5, n_samples)  # Skewed towards lower scores
    video_scores[video_true == 1] += np.random.normal(0.3, 0.2, sum(video_true == 1))
    video_scores = np.clip(video_scores, 0, 1)
    video_pred = (video_scores > 0.5).astype(int)
    
    # Audio modality (spoofing detection)
    audio_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    audio_scores = np.random.beta(2, 3, n_samples)
    audio_scores[audio_true == 1] += np.random.normal(0.4, 0.15, sum(audio_true == 1))
    audio_scores = np.clip(audio_scores, 0, 1)
    audio_pred = (audio_scores > 0.5).astype(int)
    
    # Text modality (injection detection)
    text_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    text_scores = np.random.beta(1, 4, n_samples)
    text_scores[text_true == 1] += np.random.normal(0.5, 0.2, sum(text_true == 1))
    text_scores = np.clip(text_scores, 0, 1)
    text_pred = (text_scores > 0.5).astype(int)
    
    # Fusion (combine all modalities)
    # For samples where all modalities are available
    fusion_mask = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    fusion_indices = np.where(fusion_mask)[0]
    
    fusion_true = video_true[fusion_indices]
    fusion_scores = (0.4 * video_scores[fusion_indices] + 
                    0.3 * audio_scores[fusion_indices] + 
                    0.3 * text_scores[fusion_indices])
    fusion_pred = (fusion_scores > 0.5).astype(int)
    
    return {
        'video': {'true': video_true, 'pred': video_pred, 'scores': video_scores},
        'audio': {'true': audio_true, 'pred': audio_pred, 'scores': audio_scores},
        'text': {'true': text_true, 'pred': text_pred, 'scores': text_scores},
        'fusion': {'true': fusion_true, 'pred': fusion_pred, 'scores': fusion_scores}
    }

def main():
    """Test the evaluation metrics."""
    logger.info("Testing evaluation metrics with sample data...")
    
    # Create sample data
    data = create_sample_evaluation_data()
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Evaluate individual modalities
    individual_results = {}
    for modality in ['video', 'audio', 'text']:
        results = evaluator.evaluate_modality(
            modality,
            data[modality]['true'],
            data[modality]['pred'],
            data[modality]['scores']
        )
        individual_results[modality] = results
        
        print(f"\n{modality.capitalize()} Results:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  F1-Score: {results['f1_score']:.3f}")
        print(f"  AUC-ROC: {results.get('auc_roc', 'N/A'):.3f}")
    
    # Evaluate fusion
    fusion_results = evaluator.evaluate_fusion(
        data['fusion']['true'],
        data['fusion']['pred'],
        data['fusion']['scores'],
        individual_results
    )
    
    print(f"\nFusion Results:")
    print(f"  Accuracy: {fusion_results['accuracy']:.3f}")
    print(f"  F1-Score: {fusion_results['f1_score']:.3f}")
    print(f"  AUC-ROC: {fusion_results.get('auc_roc', 'N/A'):.3f}")
    
    # Generate plots and tables
    evaluator.plot_confusion_matrices()
    
    y_true_dict = {k: v['true'] for k, v in data.items()}
    y_scores_dict = {k: v['scores'] for k, v in data.items()}
    evaluator.plot_roc_curves(y_true_dict, y_scores_dict)
    
    evaluator.plot_performance_comparison()
    
    results_df = evaluator.generate_results_table()
    print(f"\nResults Table:")
    print(results_df.to_string(index=False))
    
    evaluator.save_results()
    
    print(f"\nEvaluation complete! Results saved to experiments/results/")

if __name__ == '__main__':
    main()

