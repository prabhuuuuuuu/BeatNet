# models.py: Model architectures for CNN and LSTM.
# Why two models? To compare image-based (CNN) vs. sequence-based (LSTM) approaches for audio.
# Updates: Dynamic calculation of fc1 input size in CNN to handle varying feature map sizes.

import torch  # PyTorch core for tensors.
import torch.nn as nn  # Neural network modules.
import yaml  # For loading config.

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class GenreCNN(nn.Module):
    # CNN model for Mel spectrograms.
    def __init__(self, input_shape, num_classes, dropout):
        super(GenreCNN, self).__init__()
        
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Dynamically compute the input size for fc1 using a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # Dummy tensor with batch size 1.
            x = self.pool(nn.functional.relu(self.bn1(self.conv1(dummy))))  # Pass through conv1, bn1, relu, pool.
            x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))  # Pass through conv2, bn2, relu, pool.
            self.num_features = x.view(1, -1).size(1)  # Flatten and get the feature count.
        
        self.fc1 = nn.Linear(self.num_features, 128)  # Fully connected layer.
        self.fc2 = nn.Linear(128, num_classes)  # Output layer.
        
        print(f"Debug: Computed fc1 input features: {self.num_features}")
    
    def forward(self, x):
        # Forward pass with correct sequence.
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten.
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GenreLSTM(nn.Module):
    # LSTM model for MFCC sequences.
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(GenreLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional.
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Get final hidden states.
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Concat bidirectional hiddens.
        x = self.dropout(h_n)
        x = self.fc(x)
        return x
