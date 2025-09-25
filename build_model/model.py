import torch
import torch.nn as nn
import timm

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

