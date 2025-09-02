import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import sys
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.metrics import classification_report
from logging_config import setup_logging, get_logger, PerformanceLogger, DataLogger
from collections import Counter

# to augment data later
import albumentations as A

# Define augmentation pipeline
default_augmentation = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.HorizontalFlip(p=0.4),
    ToTensorV2()
])
#augmentations for classes without much data
heavy_augmentation = A.Compose([
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
    def __init__(self, num_classes):
        #initliazes this class with everything from nn.Module parent class
        super(DermatologyClassifier, self).__init__()
        #creates base model using timm library for image classficiation
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        #removes last layer to be replaced by our specific number of classes
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        #makes a classifier layer
        self.classifier = nn.Linear(enet_out_size, num_classes)
        #40% chance of inputs being turned to 0
        #prevents overfitting and adapting to training data
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        #connects parts and returns output
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.classifier(x)
        return output
transform = transforms.Compose([
    transforms.ToTensor()
])  
#based on available photos uses heavy augmentation (smaller datasets) 
# or default augmentation
class_weights = {0: "heavy", 1: "default", 2:"heavy", 
                 3:"default", 4:"default", 5:"default", 6:"heavy"}
#words in classweights correspond to the augmentors
transform_dict = {"default": default_augmentation, "heavy": heavy_augmentation}
# Initialize logging
logger = setup_logging(log_level="INFO")
data_logger = DataLogger(logger)
perf_logger = PerformanceLogger(logger)

logger.info("Initializing training pipeline")

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
train_dataset = DermatologyDatasetTrain(dataset_path=train_folder, class_weights=class_weights, transform_dict=transform_dict)
test_dataset = DermatologyDataset(dataset_path=test_folder, transform=transform)
val_dataset = DermatologyDataset(dataset_path=val_folder, transform=transform)

# Log dataset information
class_counts = {}
for i, class_name in enumerate(train_dataset.classes()):
    class_counts[class_name] = len([f for f in os.listdir(os.path.join(train_folder, class_name)) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
data_logger.log_dataset_info(train_folder, class_counts)

# Create data loaders
logger.info("Creating data loaders")
# Assuming train_dataset is an instance of DermatologyDatasetTrain
targets = [label for _, label in train_dataset.data.imgs]  # or train_dataset.data.samples
class_counts = Counter(targets)

num_samples = len(train_dataset)
num_classes = len(class_counts)
class_weights = {cls: num_samples / (num_classes * count) for cls, count in class_counts.items()}

sample_weights = [class_weights[label] for _, label in train_dataset.data.imgs]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # or a multiple for more epochs per epoch
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    # shuffle=False,  # Do not use shuffle with sampler
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
logger.info("Initializing model")
model = DermatologyClassifier(num_classes=7)
# Loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Scheduler with tuned patience/factor
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5, min_lr=1e-6
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
#if there is a preexisting model
#if(os.path.exists(os.path.join(os.getcwd(), "best_model_overfit.pth"))) and (os.path.exists(os.path.join(os.getcwd(), "best_optimizer_overfit.pth"))):
    #model.load_state_dict(torch.load(os.path.join(os.getcwd(), "best_model_overfit.pth"), weights_only=True))
    #optimizer.load_state_dict(torch.load(os.path.join(os.getcwd(), "best_optimizer_overfit.pth"), weights_only=True))
    #print("Loaded preexisting local model and optimizer", file=sys.stderr)
# Training configuration
num_epochs = 20
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 5
no_improve = 0


logger.info(f"Starting training for {num_epochs} epochs")
perf_logger.start_timer("total_training")

import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    perf_logger.start_timer(f"epoch_{epoch+1}_training")
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        if batch_idx % 50 == 0:  # Log progress every 50 batches
            logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    perf_logger.end_timer(f"epoch_{epoch+1}_training", f"- Train Loss: {train_loss:.4f}")
    
    # Validation phase
    model.eval()
    running_loss, correct_val = 0.0, 0.0
    y_true, y_pred = [], []
    
    perf_logger.start_timer(f"epoch_{epoch+1}_validation")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct_val += (outputs.argmax(1) == labels).sum().item()
    
    val_loss = running_loss / len(val_loader.dataset)
    accuracy = correct_val / len(val_loader.dataset)
    scheduler.step(val_loss)
    val_losses.append(val_loss)
    
    perf_logger.end_timer(f"epoch_{epoch+1}_validation", f"- Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Log epoch metrics
    data_logger.log_model_metrics(epoch+1, train_loss, val_loss, accuracy)
    
    # Model saving and early stopping
    if val_loss < best_val_loss:
        name = "best_model_overfit_new3.pth"
        opt_name = "best_optimizer_overfit_new3.pth"
        best_val_loss = val_loss
        torch.save(model.state_dict(), name)
        torch.save(optimizer.state_dict(), opt_name)
        logger.info(f"New best model saved: {name} (Val Loss: {val_loss:.4f})")
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