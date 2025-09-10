#!/usr/bin/env python3
"""
Comprehensive test evaluation script for DermaHelper model.
Evaluates the best trained model on the test dataset with detailed metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from logging_config import setup_logging, get_logger, PerformanceLogger, DataLogger
import json
from datetime import datetime

# Import the model architecture from training.py
class DermatologyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DermatologyClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.classifier(x)
        return output

class DermatologyDataset(nn.Module):
    """Simple dataset wrapper for test data (no augmentation needed)."""
    def __init__(self, dataset_path, transform=None):
        self.data = ImageFolder(dataset_path, transform=transform)
         
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def classes(self):
        return self.data.classes

def load_best_model(model_path, optimizer_path, device, num_classes=7):
    """Load the best trained model and optimizer."""
    logger = get_logger("test_evaluation")
    
    # Initialize model
    model = DermatologyClassifier(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Load model state
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded model from: {model_path}")
    else:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load optimizer state (optional)
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        logger.info(f"Loaded optimizer from: {optimizer_path}")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, optimizer

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test dataset and return comprehensive metrics."""
    logger = get_logger("test_evaluation")
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting model evaluation on test dataset")
    perf_logger.start_timer("test_evaluation")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    perf_logger.end_timer("test_evaluation", f"- Processed {len(all_predictions)} test samples")
    
    return all_predictions, all_labels, all_probabilities

def generate_classification_report(predictions, labels, class_names):
    """Generate detailed classification report."""
    logger = get_logger("test_evaluation")
    
    # Calculate overall accuracy
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate classification report
    report = classification_report(
        labels, 
        predictions, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Log per-class metrics
    logger.info("Per-class Performance:")
    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            logger.info(f"  {class_name:12}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
    
    # Log macro averages
    macro_avg = report['macro avg']
    logger.info(f"Macro Average: Precision={macro_avg['precision']:.3f}, Recall={macro_avg['recall']:.3f}, F1={macro_avg['f1-score']:.3f}")
    
    return report, accuracy

def create_confusion_matrix(predictions, labels, class_names, save_path=None):
    """Create and save confusion matrix visualization."""
    logger = get_logger("test_evaluation")
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples'}
    )
    plt.title('Confusion Matrix - Test Dataset Evaluation', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm

def save_test_results(report, accuracy, confusion_matrix, class_names, model_path, save_dir="test_results"):
    """Save comprehensive test results to files."""
    logger = get_logger("test_evaluation")
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results = {
        "model_path": model_path,
        "timestamp": timestamp,
        "overall_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": class_names
    }
    
    results_file = os.path.join(save_dir, f"test_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Save summary report
    summary_file = os.path.join(save_dir, f"test_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("DermaHelper Test Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("Per-class Performance:\n")
        f.write("-" * 30 + "\n")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                f.write(f"{class_name:12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}\n")
        
        f.write(f"\nMacro Average: P={report['macro avg']['precision']:.3f}, R={report['macro avg']['recall']:.3f}, F1={report['macro avg']['f1-score']:.3f}\n")
    
    logger.info(f"Summary report saved to: {summary_file}")
    
    return results_file, summary_file

def main():
    """Main function to run comprehensive test evaluation."""
    # Initialize logging
    logger = setup_logging(log_level="INFO")
    data_logger = DataLogger(logger)
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting DermaHelper Test Evaluation")
    perf_logger.start_timer("total_test_evaluation")
    
    # Configuration
    num_classes = 7
    batch_size = 32
    
    # Model paths (using the best model as noted in models/notes.txt)
    models_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(models_dir, "best_model_0902_001920.pth")
    optimizer_path = os.path.join(models_dir, "best_optimizer_0902_001920.pth")
    
    # Data paths
    test_folder = os.path.join(os.getcwd(), "usable_data", "test")
    
    # Validate paths
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Available model files:")
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                logger.info(f"  - {file}")
        return
    
    if not os.path.exists(test_folder):
        logger.error(f"Test data folder not found: {test_folder}")
        return
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) for evaluation")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for evaluation")
    
    # Load model
    logger.info("Loading best trained model")
    model, optimizer = load_best_model(model_path, optimizer_path, device, num_classes)
    
    # Create test dataset and loader
    logger.info("Creating test dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_dataset = DermatologyDataset(dataset_path=test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get class names
    class_names = test_dataset.classes()
    logger.info(f"Test dataset classes: {class_names}")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    
    # Log dataset information
    class_counts = {}
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(test_folder, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
    data_logger.log_dataset_info(test_folder, class_counts)
    
    # Evaluate model
    logger.info("Evaluating model on test dataset")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device, class_names)
    
    # Generate classification report
    logger.info("Generating classification report")
    report, accuracy = generate_classification_report(predictions, labels, class_names)
    
    # Create confusion matrix
    logger.info("Creating confusion matrix")
    cm_path = os.path.join("test_results", f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    os.makedirs("test_results", exist_ok=True)
    confusion_matrix = create_confusion_matrix(predictions, labels, class_names, cm_path)
    
    # Save results
    logger.info("Saving test results")
    results_file, summary_file = save_test_results(report, accuracy, confusion_matrix, class_names, model_path)
    
    perf_logger.end_timer("total_test_evaluation", f"- Final Test Accuracy: {accuracy:.4f}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TEST EVALUATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Model: {os.path.basename(model_path)}")
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Test Samples: {len(predictions)}")
    logger.info(f"Results saved to: test_results/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

