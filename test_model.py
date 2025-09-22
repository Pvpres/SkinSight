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
import argparse
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from logging_config import setup_logging, get_logger, PerformanceLogger, DataLogger
import json
from datetime import datetime

# Import the model architecture from training.py (updated for two-head)
class DermatologyClassifier(nn.Module):
    def __init__(self, num_classes, use_two_heads=True):
        super(DermatologyClassifier, self).__init__()
        self.use_two_heads = use_two_heads
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        
        if use_two_heads:
            # Two-head architecture
            # Match training architecture naming to load weights correctly
            self.shared_fc = nn.Linear(enet_out_size, 512)
            self.binary_head = nn.Linear(512, 2)  # healthy vs unhealthy
            self.disease_head = nn.Linear(512, 4)  # 4 disease types
        else:
            # Single head (legacy)
            self.classifier = nn.Linear(enet_out_size, num_classes)
        
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        if self.use_two_heads:
            # Match training forward: apply ReLU after shared_fc
            shared = torch.relu(self.shared_fc(x))
            binary_output = self.binary_head(shared)
            disease_output = self.disease_head(shared)
            return binary_output, disease_output
        else:
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

# Two-head model helper functions
def prepare_labels_for_two_head(labels):
    """Convert 5-class labels to binary + disease labels"""
    # Binary mapping: Class 3 (healthy) = 0, all others = unhealthy (1)
    binary_labels = (labels != 3).long()  # Class 3 (healthy) = 0, others = 1
    
    # Disease mapping for unhealthy classes: 0,1,2,4 → 0,1,2,3
    # acne(0)→0, dry(1)→1, eczema(2)→2, oily(4)→3
    disease_mapping = torch.tensor([0, 1, 2, -1, 3], device=labels.device)  # -1 for healthy (ignored)
    disease_labels = disease_mapping[labels]
    
    return binary_labels, disease_labels

def combine_predictions_to_original(binary_pred, disease_pred):
    """
    Convert binary + disease predictions back to original 5-class format
    - Binary 0 (healthy) → Class 3 (healthy)
    - Binary 1 + Disease 0 → Class 0 (acne)
    - Binary 1 + Disease 1 → Class 1 (dry)
    - Binary 1 + Disease 2 → Class 2 (eczema)
    - Binary 1 + Disease 3 → Class 4 (oily)
    """
    disease_to_original = torch.tensor([0, 1, 2, 4], device=disease_pred.device)
    mapped_disease = disease_to_original[disease_pred]
    combined_pred = torch.where(binary_pred == 0, 3, mapped_disease)
    return combined_pred

def load_best_model(model_path, optimizer_path, device, num_classes=5, use_two_heads=True):
    """Load the best trained model and optimizer."""
    logger = get_logger("test_evaluation")
    
    # Initialize model
    model = DermatologyClassifier(num_classes=num_classes, use_two_heads=use_two_heads)
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
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
            logger.info(f"Loaded optimizer from: {optimizer_path}")
        except ValueError as e:
            # Optimizer param groups may differ (e.g., training used multiple groups). Not needed for eval.
            logger.warning(f"Optimizer state not loaded due to mismatch ({e}). Proceeding with fresh optimizer for evaluation.")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state ({e}). Proceeding with fresh optimizer for evaluation.")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, optimizer

def evaluate_model(model, test_loader, device, class_names, use_two_heads=True):
    """Evaluate the model on test dataset and return comprehensive metrics."""
    logger = get_logger("test_evaluation")
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting model evaluation on test dataset")
    if use_two_heads:
        logger.info("Using two-head model evaluation")
    perf_logger.start_timer("test_evaluation")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_binary_predictions = []
    all_disease_predictions = []
    all_binary_labels = []
    all_disease_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            if use_two_heads:
                # Two-head model evaluation
                binary_output, disease_output = model(images)
                
                # Get predictions
                binary_pred = torch.argmax(binary_output, dim=1)
                disease_pred = torch.argmax(disease_output, dim=1)
                
                # Combine predictions to original format
                combined_pred = combine_predictions_to_original(binary_pred, disease_pred)
                
                # Prepare labels for two-head
                binary_labels, disease_labels = prepare_labels_for_two_head(labels)
                
                # Store results
                all_predictions.extend(combined_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_binary_predictions.extend(binary_pred.cpu().numpy())
                all_disease_predictions.extend(disease_pred.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
                all_disease_labels.extend(disease_labels.cpu().numpy())
                
                # Calculate probabilities for combined predictions
                binary_probs = torch.softmax(binary_output, dim=1)
                disease_probs = torch.softmax(disease_output, dim=1)
                
                # For simplicity, use binary probabilities as main probabilities
                all_probabilities.extend(binary_probs.cpu().numpy())
                
            else:
                # Single-head model evaluation (legacy)
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
    
    if use_two_heads:
        return (all_predictions, all_labels, all_probabilities, 
                all_binary_predictions, all_disease_predictions,
                all_binary_labels, all_disease_labels)
    else:
        return all_predictions, all_labels, all_probabilities

def generate_classification_report(predictions, labels, class_names, use_two_heads=True, 
                                 binary_predictions=None, disease_predictions=None,
                                 binary_labels=None, disease_labels=None):
    """Generate detailed classification report."""
    logger = get_logger("test_evaluation")
    
    # Calculate overall accuracy
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if use_two_heads and binary_predictions is not None:
        # Calculate binary and disease accuracies
        binary_accuracy = accuracy_score(binary_labels, binary_predictions)
        logger.info(f"Binary Classification Accuracy: {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
        
        # Disease accuracy (only for unhealthy samples)
        unhealthy_mask = np.array(binary_labels) == 1
        if np.sum(unhealthy_mask) > 0:
            disease_accuracy = accuracy_score(np.array(disease_labels)[unhealthy_mask], 
                                            np.array(disease_predictions)[unhealthy_mask])
            logger.info(f"Disease Classification Accuracy: {disease_accuracy:.4f} ({disease_accuracy*100:.2f}%)")
            logger.info(f"Disease samples evaluated: {np.sum(unhealthy_mask)}")
        else:
            logger.info("No unhealthy samples found for disease classification")
    
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

def find_most_recent_model(models_dir):
    """Find the most recent model file in the models directory."""
    logger = get_logger("test_evaluation")
    
    # Look for all model files
    model_pattern = os.path.join(models_dir, "best_model_*.pth")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return None
    
    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    most_recent = model_files[0]
    
    logger.info(f"Found {len(model_files)} model files")
    logger.info(f"Most recent model: {os.path.basename(most_recent)}")
    
    return most_recent

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test DermaHelper model on test dataset')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to specific model to test (e.g., models/best_model_0902_001920.pth). If not specified, uses most recent model.')
    parser.add_argument('--list-models', action='store_true', 
                        help='List all available models and exit')
    parser.add_argument('--single-head', action='store_true', 
                        help='Use single-head model (legacy mode) instead of two-head')
    args = parser.parse_args()
    
    # Initialize logging
    logger = setup_logging(log_level="INFO")
    data_logger = DataLogger(logger)
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting DermaHelper Test Evaluation")
    perf_logger.start_timer("total_test_evaluation")
    
    # Configuration
    num_classes = 5  # Updated for 5-class dataset
    batch_size = 32
    use_two_heads = not args.single_head  # Use two-head model unless --single-head is specified
    
    # Model paths
    models_dir = os.path.join(os.getcwd(), "models")
    
    # Handle list models option
    if args.list_models:
        model_files = glob.glob(os.path.join(models_dir, "best_model_*.pth"))
        if model_files:
            print("Available models:")
            model_files.sort(key=os.path.getmtime, reverse=True)
            for i, model_file in enumerate(model_files, 1):
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                print(f"  {i}. {os.path.basename(model_file)} (modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("No model files found in models/ directory")
        return
    
    # Determine which model to use
    if args.model:
        model_path = args.model
        logger.info(f"Using specified model: {model_path}")
    else:
        model_path = find_most_recent_model(models_dir)
        if not model_path:
            logger.error("No model files found and no model specified")
            return
        logger.info(f"Using most recent model: {model_path}")
    
    # Try to find corresponding optimizer
    optimizer_path = model_path.replace("best_model_", "best_optimizer_")
    
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
    logger.info("Loading best trained two-head model")
    model, optimizer = load_best_model(model_path, optimizer_path, device, num_classes, use_two_heads)
    
    # Create test dataset and loader
    logger.info("Creating test dataset")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure all images are 224x224
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
    logger.info("Evaluating two-head model on test dataset")
    if use_two_heads:
        (predictions, labels, probabilities, 
         binary_predictions, disease_predictions,
         binary_labels, disease_labels) = evaluate_model(model, test_loader, device, class_names, use_two_heads)
    else:
        predictions, labels, probabilities = evaluate_model(model, test_loader, device, class_names, use_two_heads)
    
    # Generate classification report
    logger.info("Generating classification report")
    if use_two_heads:
        report, accuracy = generate_classification_report(
            predictions, labels, class_names, use_two_heads,
            binary_predictions, disease_predictions,
            binary_labels, disease_labels
        )
    else:
        report, accuracy = generate_classification_report(predictions, labels, class_names, use_two_heads)
    
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
    
    # Print usage examples
    print("\n" + "="*60)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Usage examples for future testing:")
    print("  Test most recent (two-head):    python test_model.py")
    print("  Test specific model:            python test_model.py --model models/best_model_twohead_0918_021300.pth")
    print("  Test single-head model:         python test_model.py --single-head")
    print("  List all models:                python test_model.py --list-models")
    print("="*60)

if __name__ == "__main__":
    main()

