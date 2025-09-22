import torch, numpy as np, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import argparse
import math
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import numpy as np
from logging_config import setup_logging, PerformanceLogger, DataLogger
from collections import Counter
import datetime
# to augment data later
import albumentations as A

# Define augmentation pipeline with consistent resizing
default_augmentation = A.Compose([
    A.Resize(224, 224),  # Ensure all images are 224x224
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.HorizontalFlip(p=0.4),
    ToTensorV2()
])
#augmentations for classes without much data
heavy_augmentation = A.Compose([
    A.Resize(224, 224),  # Ensure all images are 224x224
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(limit=45, p=0.8),
    A.HorizontalFlip(p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.7),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.7),
    ToTensorV2()
])
#creates torchvision dataset object
#DermatologyDataset(Dataset) tells class to inherit properties from Dataset
class DermatologyDatasetTrain(Dataset):
    #initializes dataset with dataset wrapper called ImageFolder
    def __init__(self, dataset_path, class_weights=None, transform_dict=None):
        self.data = ImageFolder(dataset_path)
        self.class_weights = class_weights
        self.transform_dict = transform_dict

    #tells len what to return which is length of dataset
    def __len__(self):
        return len(self.data)
    #returns image at index
    #checks what group image is in and add the corresponding transformation
    #returns image
    def __getitem__(self, index):
        image, label = self.data[index] #load image and label
        transform_type = self.class_weights[label] #get augmentation type
        transform = self.transform_dict[transform_type] #get specific transformation
        image = np.array(image) #turns pil image to numpy
        transformed = transform(image=image)
        image = transformed["image"]
        return image, label
    
    def classes(self):
        return self.data.classes

#for testing and validation data
#does not do any transformations
class DermatologyDataset(Dataset):
    #initializes dataset with dataset wrapper called ImageFolder
    def __init__(self, dataset_path,transform=None):
        self.data = ImageFolder(dataset_path, transform=transform)
         
    #tells len what to return which is length of dataset
    def __len__(self):
        return len(self.data)
    #returns image at index
    def __getitem__(self, index):
        return self.data[index]
    
    def classes(self):
        return self.data.classes

class DermatologyClassifier(nn.Module):
    def __init__(self, num_classes, use_two_heads=True):
        #determines whether to use two heads one binary and other multiclass
        self.use_two_heads = use_two_heads
        #initliazes this class with everything from nn.Module parent class
        super(DermatologyClassifier, self).__init__()
        #creates base model using timm library for image classficiation
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        #removes last layer to be replaced by our specific number of classes
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        #40% chance of inputs being turned to 0
        #prevents overfitting and adapting to training data
        self.dropout = nn.Dropout(p=0.4)
        
        if use_two_heads:
            # Two-head architecture - REPLACES the single classifier
            self.shared_fc = nn.Linear(enet_out_size, 512)
            self.binary_head = nn.Linear(512, 2)  # healthy vs unhealthy
            self.disease_head = nn.Linear(512, 4)  # 4 disease types
            print("Two-head classifier initialized: Binary (2) + Disease (4)")
        else:
            # Original single head - KEEPS your original architecture
            self.classifier = nn.Linear(enet_out_size, num_classes)  # <-- THIS LINE IS STILL HERE!
            print(f"Single-head classifier initialized: {num_classes} classes")
    def forward(self, x):
        #connects parts and returns output
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        if self.use_two_heads:
            shared = torch.relu(self.shared_fc(x))
            binary_output = self.binary_head(shared)
            disease_output = self.disease_head(shared)
            return binary_output, disease_output
        else:
            # Original path: direct to classifier (your original code)
            return self.classifier(x)
#======================CLASSIFIER HELPERS======================
def prepare_labels_for_two_head(labels):
    """Convert 5-class labels to binary + disease labels"""
    # Binary mapping: Class 3 (healthy) = 0, all others = unhealthy (1)
    binary_labels = (labels != 3).long()  # Class 3 (healthy) = 0, others = 1
    
    # Disease mapping for unhealthy classes: 0,1,2,4 → 0,1,2,3
    # acne(0)→0, dry(1)→1, eczema(2)→2, oily(4)→3
    disease_mapping = torch.tensor([0, 1, 2, -1, 3], device=labels.device)  # -1 for healthy (ignored)
    disease_labels = disease_mapping[labels]
    
    return binary_labels, disease_labels

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by focusing on hard examples"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_two_head_loss(binary_output, disease_output, binary_labels, disease_labels, 
                           binary_criterion, disease_criterion, binary_weight=1.5, disease_weight=1.0,
                           use_focal_loss=True, focal_alpha=1, focal_gamma=2):
    """Calculate combined loss for two heads with focal loss for hard examples"""
    
    if use_focal_loss:
        # Use Focal Loss for better handling of hard examples
        focal_binary = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        focal_disease = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Binary loss (all samples)
        binary_loss = focal_binary(binary_output, binary_labels)
        
        # Disease loss (only unhealthy samples)
        unhealthy_mask = binary_labels == 1
        if unhealthy_mask.sum() > 0:
            disease_loss = focal_disease(disease_output[unhealthy_mask], disease_labels[unhealthy_mask])
        else:
            disease_loss = torch.tensor(0.0, device=binary_output.device)
    else:
        # Standard weighted loss
        binary_loss = binary_criterion(binary_output, binary_labels)
        
        # Disease loss (only unhealthy samples)
        unhealthy_mask = binary_labels == 1
        if unhealthy_mask.sum() > 0:
            disease_loss = disease_criterion(disease_output[unhealthy_mask], disease_labels[unhealthy_mask])
        else:
            disease_loss = torch.tensor(0.0, device=binary_output.device)
    
    # Optimized weighting: Binary is more important for overall accuracy
    total_loss = binary_weight * binary_loss + disease_weight * disease_loss
    return total_loss, binary_loss, disease_loss

def combine_predictions_to_original(binary_pred, disease_pred):
    """
    Convert binary + disease predictions back to original 5-class format
    
    Reverse mapping:
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are 224x224
    transforms.ToTensor()
])  

# Class-specific augmentation weights for 5-class dataset
# Maps class index to augmentation strategy based on dataset size
augmentation_weights = {
    0: "heavy",   # acne (980 samples) - use heavy augmentation for variety
    1: "heavy",   # dry (261 samples) - use heavy augmentation (smaller dataset)
    2: "heavy",   # eczema (196 samples) - use heavy augmentation (smaller dataset)
    3: "default", # healthy (3947 samples) - use default augmentation (largest dataset)
    4: "heavy"    # oily (417 samples) - use heavy augmentation for variety
}
#words in classweights correspond to the augmentors
transform_dict = {"default": default_augmentation, "heavy": heavy_augmentation}
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train DermaHelper model')
parser.add_argument('--resume', type=str, default=None, 
                    help='Path to model checkpoint to resume training from (e.g., models/best_model_0902_001920.pth)')
parser.add_argument('--epochs', type=int, default=20, 
                    help='Number of epochs to train (default: 20)')
args = parser.parse_args()

# Initialize logging
logger = setup_logging(log_level="INFO")
data_logger = DataLogger(logger)
perf_logger = PerformanceLogger(logger)

logger.info("Initializing training pipeline")
if args.resume:
    logger.info(f"Resuming training from: {args.resume}")
else:
    logger.info("Starting fresh training")

# Ensure models directory exists for saving checkpoints
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
logger.info(f"Model save directory: {models_dir}")

# Create paths to data on local device
train_folder = os.path.join(os.getcwd(), "usable_data", "train")
test_folder = os.path.join(os.getcwd(), "usable_data", "test")
val_folder = os.path.join(os.getcwd(), "usable_data", "val")

# Validate data paths
for folder, name in [(train_folder, "train"), (test_folder, "test"), (val_folder, "val")]:
    if not os.path.exists(folder):
        logger.error(f"Data folder not found: {folder}")
        raise FileNotFoundError(f"Required data folder not found: {name}")

logger.info("Data paths validated successfully")

# Create dataset objects
logger.info("Creating dataset objects")
train_dataset = DermatologyDatasetTrain(dataset_path=train_folder, class_weights=augmentation_weights, transform_dict=transform_dict)
test_dataset = DermatologyDataset(dataset_path=test_folder, transform=transform)
val_dataset = DermatologyDataset(dataset_path=val_folder, transform=transform)

# Log dataset information
class_counts = {}
for i, class_name in enumerate(train_dataset.classes()):
    class_counts[class_name] = len([f for f in os.listdir(os.path.join(train_folder, class_name)) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
data_logger.log_dataset_info(train_folder, class_counts)

# Log updated acne counts specifically
logger.info("Updated dataset with additional acne images:")
for split_name, split_path in [("Train", train_folder), ("Val", val_folder), ("Test", test_folder)]:
    acne_path = os.path.join(split_path, "acne")
    if os.path.exists(acne_path):
        acne_count = len([f for f in os.listdir(acne_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        logger.info(f"  {split_name}: {acne_count} acne images")

# Create data loaders with comprehensive class imbalance strategies
logger.info("Creating data loaders with class imbalance handling")

# Get class distribution
targets = [label for _, label in train_dataset.data.imgs]
class_counts = Counter(targets)
num_samples = len(train_dataset)
num_classes = len(class_counts)

# Log detailed class distribution
logger.info("Class distribution analysis:")
for i, class_name in enumerate(train_dataset.classes()):
    count = class_counts.get(i, 0)
    percentage = (count / num_samples) * 100
    logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")

# Calculate inverse frequency weights (more aggressive for severe imbalance)
max_count = max(class_counts.values())
class_weights = {}
for cls in range(num_classes):
    count = class_counts.get(cls, 1)  # Avoid division by zero
    # Use sqrt of inverse frequency for less aggressive weighting
    class_weights[cls] = math.sqrt(max_count / count)

logger.info("Class weights for sampling:")
for i, class_name in enumerate(train_dataset.classes()):
    logger.info(f"  {class_name}: {class_weights.get(i, 1.0):.3f}")

# Create sample weights for WeightedRandomSampler
sample_weights = [class_weights[label] for _, label in train_dataset.data.imgs]

# Use more aggressive oversampling (2x the dataset size)
oversample_factor = 2
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights) * oversample_factor,
    replacement=True
)

# Implement balanced batch sampling for more equal representation
class BalancedBatchSampler:
    """Custom sampler to ensure more balanced batches"""
    def __init__(self, dataset, batch_size=32, oversample_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.oversample_factor = oversample_factor
        
        # Group samples by class
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset.data.imgs):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // len(self.class_indices))
        
        # Pre-generate all indices for the sampler
        self.all_indices = []
        for _ in range(self.oversample_factor):
            for _ in range(len(self.dataset) // self.batch_size):
                batch_indices = []
                
                # Sample from each class
                for class_idx, indices in self.class_indices.items():
                    # Sample with replacement if needed
                    class_samples = np.random.choice(indices, size=self.samples_per_class, replace=True)
                    batch_indices.extend(class_samples)
                
                # Fill remaining slots randomly
                remaining = self.batch_size - len(batch_indices)
                if remaining > 0:
                    all_indices = list(range(len(self.dataset)))
                    additional = np.random.choice(all_indices, size=remaining, replace=True)
                    batch_indices.extend(additional)
                
                self.all_indices.extend(batch_indices[:self.batch_size])
        
    def __iter__(self):
        # Yield individual indices, not batches
        for idx in self.all_indices:
            yield idx
    
    def __len__(self):
        return len(self.all_indices)

# Use the standard WeightedRandomSampler (more reliable)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    # shuffle=False,  # Do not use shuffle with sampler
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
logger.info("Initializing two-head model")
model = DermatologyClassifier(num_classes=5, use_two_heads=True)

# Calculate loss weights for class imbalance
def calculate_loss_weights(class_counts, device):
    """Calculate loss weights to penalize minority class misclassifications more"""
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Calculate inverse frequency weights for loss
    loss_weights = torch.zeros(num_classes, device=device)
    for cls, count in class_counts.items():
        if count > 0:
            loss_weights[cls] = total_samples / (num_classes * count)
    
    # Normalize weights to prevent exploding gradients
    loss_weights = loss_weights / loss_weights.mean()
    
    return loss_weights

# Binary loss weights (healthy vs unhealthy)
binary_class_counts = {0: class_counts.get(3, 0), 1: sum(class_counts[i] for i in range(5) if i != 3)}
binary_loss_weights = calculate_loss_weights(binary_class_counts, torch.device("cpu"))

# Disease loss weights (4 disease classes: 0,1,2,4 -> 0,1,2,3)
disease_class_counts = {}
disease_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map original indices to disease head indices
for orig_idx, new_idx in disease_mapping.items():
    disease_class_counts[new_idx] = class_counts.get(orig_idx, 0)

disease_loss_weights = calculate_loss_weights(disease_class_counts, torch.device("cpu"))

logger.info("Loss weights calculated:")
logger.info(f"Binary weights: {binary_loss_weights}")
logger.info(f"Disease weights: {disease_loss_weights}")

# Loss functions with class weights (will be moved to device later)
binary_criterion = nn.CrossEntropyLoss(weight=binary_loss_weights, label_smoothing=0.1)
disease_criterion = nn.CrossEntropyLoss(weight=disease_loss_weights, label_smoothing=0.15)
# Keep your original criterion as fallback
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Implement class-specific learning rates for better minority class learning
def get_class_specific_optimizer(model, class_counts, base_lr=0.001):
    """Create optimizer with different learning rates for different parts of the model"""
    # Higher learning rate for disease head (minority classes)
    # Lower learning rate for binary head (majority class)
    disease_head_params = []
    binary_head_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if 'disease_head' in name:
            disease_head_params.append(param)
        elif 'binary_head' in name:
            binary_head_params.append(param)
        else:
            base_params.append(param)
    
    # Calculate learning rate multipliers based on class imbalance
    max_count = max(class_counts.values())
    minority_classes = [cls for cls, count in class_counts.items() if count < max_count * 0.3]
    
    # Higher LR for disease head to learn minority classes better
    disease_lr = base_lr * 2.0  # 2x for disease classification
    binary_lr = base_lr * 0.5   # 0.5x for binary (healthy is majority)
    base_lr = base_lr * 1.0     # Standard for base model
    
    param_groups = [
        {'params': base_params, 'lr': base_lr},
        {'params': binary_head_params, 'lr': binary_lr},
        {'params': disease_head_params, 'lr': disease_lr}
    ]
    
    logger.info(f"Class-specific learning rates: Base={base_lr}, Binary={binary_lr}, Disease={disease_lr}")
    return optim.Adam(param_groups, weight_decay=1e-4)

optimizer = get_class_specific_optimizer(model, class_counts, base_lr=0.001)

# Scheduler with tuned patience/factor for two-head model
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.7, min_lr=1e-6
)

# Setup device
if torch.backends.mps.is_available():
    logger.info("MPS backend is available and in use.")
    device = torch.device("mps")
else:
    logger.info("MPS backend is not available, using CPU. (not recommended)")
    device = torch.device("cpu")

model.to(device)
logger.info(f"Model moved to device: {device}")

# Move loss weights to device and recreate loss functions
binary_loss_weights = binary_loss_weights.to(device)
disease_loss_weights = disease_loss_weights.to(device)

# Recreate loss functions with device-specific weights
binary_criterion = nn.CrossEntropyLoss(weight=binary_loss_weights, label_smoothing=0.1)
disease_criterion = nn.CrossEntropyLoss(weight=disease_loss_weights, label_smoothing=0.15)

logger.info("Loss functions recreated with device-specific weights")

# Load existing model if specified
if args.resume:
    if os.path.exists(args.resume):
        logger.info(f"Loading model from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        
        # Try to load corresponding optimizer
        optimizer_path = args.resume.replace("best_model_", "best_optimizer_")
        if os.path.exists(optimizer_path):
            logger.info(f"Loading optimizer from: {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        else:
            logger.warning(f"Optimizer file not found: {optimizer_path}. Starting with fresh optimizer.")
        
        logger.info("Successfully loaded existing model and optimizer")
    else:
        logger.error(f"Model file not found: {args.resume}")
        logger.info("Starting with fresh model instead")
        args.resume = None
# Training configuration
num_epochs = args.epochs
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 12
no_improve = 0

if args.resume:
    logger.info(f"Resuming training for {num_epochs} epochs")
else:
    logger.info(f"Starting fresh training for {num_epochs} epochs")
perf_logger.start_timer("total_training")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    running_binary_loss = 0.0
    running_disease_loss = 0.0
    binary_correct = 0
    disease_correct = 0
    total_samples = 0
    unhealthy_samples = 0
    
    perf_logger.start_timer(f"epoch_{epoch+1}_training")
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        images, labels = images.to(device), labels.to(device)
        
        # Prepare labels for two-head training
        binary_labels, disease_labels = prepare_labels_for_two_head(labels)
        
        optimizer.zero_grad()
        
        # Forward pass - now returns two outputs
        binary_output, disease_output = model(images)
        
        # Calculate combined loss
        loss, binary_loss, disease_loss = calculate_two_head_loss(
            binary_output, disease_output, binary_labels, disease_labels,
            binary_criterion, disease_criterion, binary_weight=1.5, disease_weight=1.0,
            use_focal_loss=True, focal_alpha=1, focal_gamma=2
        )
        
        loss.backward()
        
        # Apply gradient scaling for minority classes
        # Scale gradients for disease head more aggressively
        for name, param in model.named_parameters():
            if 'disease_head' in name and param.grad is not None:
                # Scale up gradients for disease head to learn minority classes better
                param.grad *= 1.5
            elif 'binary_head' in name and param.grad is not None:
                # Scale down gradients for binary head (healthy is majority)
                param.grad *= 0.8
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        # Accumulate losses and accuracies
        running_loss += loss.item() * images.size(0)
        running_binary_loss += binary_loss.item() * images.size(0)
        running_disease_loss += disease_loss.item() * images.size(0)
        
        # Calculate accuracies
        binary_pred = torch.argmax(binary_output, dim=1)
        binary_correct += (binary_pred == binary_labels).sum().item()
        
        unhealthy_mask = binary_labels == 1
        if unhealthy_mask.sum() > 0:
            disease_pred = torch.argmax(disease_output[unhealthy_mask], dim=1)
            disease_correct += (disease_pred == disease_labels[unhealthy_mask]).sum().item()
            unhealthy_samples += unhealthy_mask.sum().item()
        
        total_samples += images.size(0)
        
        if batch_idx % 50 == 0:
            # For logging, show some sample predictions in original format
            with torch.no_grad():
                binary_pred_sample = torch.argmax(binary_output[:5], dim=1)  # First 5 samples
                disease_pred_sample = torch.argmax(disease_output[:5], dim=1)
                combined_pred_sample = combine_predictions_to_original(binary_pred_sample, disease_pred_sample)
                original_labels_sample = labels[:5]
                
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {loss.item():.4f}, Binary: {binary_loss.item():.4f}, "
                            f"Disease: {disease_loss.item():.4f}")
                logger.debug(f"Sample predictions: Original={original_labels_sample.cpu().tolist()}, "
                            f"Combined={combined_pred_sample.cpu().tolist()}")
                
                # Log class-specific performance
                class_names = train_dataset.classes()
                logger.info(f"Batch {batch_idx} - Class distribution in batch:")
                for class_idx in range(5):
                    class_count = (labels == class_idx).sum().item()
                    if class_count > 0:
                        logger.info(f"  {class_names[class_idx]}: {class_count} samples")
    # Calculate epoch metrics
    train_loss = running_loss / len(train_loader.dataset)
    binary_train_loss = running_binary_loss / len(train_loader.dataset)
    disease_train_loss = running_disease_loss / len(train_loader.dataset)
    binary_acc = binary_correct / total_samples
    disease_acc = disease_correct / unhealthy_samples if unhealthy_samples > 0 else 0
    
    train_losses.append(train_loss)
    perf_logger.end_timer(f"epoch_{epoch+1}_training", 
                         f"- Train Loss: {train_loss:.4f}, Binary Acc: {binary_acc:.4f}, Disease Acc: {disease_acc:.4f}")
    
    # ==== EDIT 5: Replace validation phase ====
    # Validation phase
    model.eval()
    running_loss = 0.0
    binary_correct_val = 0
    disease_correct_val = 0
    total_samples_val = 0
    unhealthy_samples_val = 0
    y_true, y_pred = [], []
    
    perf_logger.start_timer(f"epoch_{epoch+1}_validation")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            # Prepare labels
            binary_labels, disease_labels = prepare_labels_for_two_head(labels)
            
            # Forward pass
            binary_output, disease_output = model(images)
            
            # Calculate loss
            loss, binary_loss, disease_loss = calculate_two_head_loss(
                binary_output, disease_output, binary_labels, disease_labels,
                binary_criterion, disease_criterion, binary_weight=1.5, disease_weight=1.0
            )
            
            # For compatibility with original logic, create combined predictions
            binary_pred = torch.argmax(binary_output, dim=1)
            disease_pred = torch.argmax(disease_output, dim=1)
            
            # Combine predictions using the helper function
            combined_pred = combine_predictions_to_original(binary_pred, disease_pred)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(combined_pred.cpu().numpy())
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate individual head accuracies
            binary_correct_val += (binary_pred == binary_labels).sum().item()
            unhealthy_mask = binary_labels == 1
            if unhealthy_mask.sum() > 0:
                disease_correct_val += (disease_pred[unhealthy_mask] == disease_labels[unhealthy_mask]).sum().item()
                unhealthy_samples_val += unhealthy_mask.sum().item()
            
            total_samples_val += images.size(0)
    
    val_loss = running_loss / len(val_loader.dataset)
    
    # Overall accuracy (matches original 5-class accuracy)
    overall_accuracy = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
    binary_val_acc = binary_correct_val / total_samples_val
    disease_val_acc = disease_correct_val / unhealthy_samples_val if unhealthy_samples_val > 0 else 0
    
    scheduler.step(val_loss)
    val_losses.append(val_loss)
    
    perf_logger.end_timer(f"epoch_{epoch+1}_validation", 
                         f"- Val Loss: {val_loss:.4f}, Overall Acc: {overall_accuracy:.4f}, "
                         f"Binary Acc: {binary_val_acc:.4f}, Disease Acc: {disease_val_acc:.4f}")
    
    # Log epoch metrics (keeping original interface)
    data_logger.log_model_metrics(epoch+1, train_loss, val_loss, overall_accuracy)
    
    # Model saving (rest stays the same)
    if val_loss < best_val_loss:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        name = f"best_model_twohead_{timestamp}.pth"
        opt_name = f"best_optimizer_twohead_{timestamp}.pth"
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(models_dir, name))
        torch.save(optimizer.state_dict(), os.path.join(models_dir, opt_name))
        logger.info(f"New best two-head model saved: {name} (Val Loss: {val_loss:.4f})")
        no_improve = 0
    else:
        no_improve += 1
        logger.debug(f"No improvement for {no_improve} epochs")
    
    if no_improve >= patience:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break
    
perf_logger.end_timer("total_training", f"- Best Val Loss: {best_val_loss:.4f}")

# Final logging
logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
logger.info(f"Final training losses: {train_losses}")
logger.info(f"Final validation losses: {val_losses}")
logger.info("Training pipeline finished successfully")


# Print usage examples
print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Model: Check models/ folder for the latest best_model_twohead_*.pth file")
print("\nUsage examples for future training:")
print("  Fresh training:     python training.py")
print("  Resume training:    python training.py --resume models/best_model_0902_001920.pth")
print("  Custom epochs:      python training.py --epochs 30")
print("  Resume + epochs:    python training.py --resume models/best_model_0902_001920.pth --epochs 10")
print("="*60)