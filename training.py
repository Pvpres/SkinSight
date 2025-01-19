import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# to augment data later
import albumentations as A

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
])

# Apply augmentation to an image
#augmented_image = augmentation_pipeline(image=resized_face)["image"]

#creates torchvision dataset object
#DermatologyDataset(Dataset) tells class to inherit properties from Dataset
class DermatologyDataset(Dataset):
    #initializes dataset with dataset wrapper called ImageFolder
    def __init__(self, dataset_path, transform=None):
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
    def forward(self, x):
        #connects parts and returns output
        x = self.features(x)
        output = self.classifier(x)
        return output

#creates path to data on local device
path = os.path.join(os.getcwd(), "usable_data")
transform = transforms.Compose([
    transforms.ToTensor()
])
#creates dataset object of images
dataset = DermatologyDataset(dataset_path=path, transform=transform)
#batch size is how many images to process at once
#shuffle tells dataloader to randomize the images it pulls
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = DermatologyClassifier(num_classes=7)
#one of built in loss functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for images, labels in dataloader:
    print(criterion(model(images), labels))
    break

