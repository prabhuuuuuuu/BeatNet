import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models import GenreCNN, GenreLSTM
from data import get_dataloaders
import os

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def train_model(model_type='cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, _, _ = get_dataloaders(model_type)
    
    if model_type == 'cnn':
        model = GenreCNN(config['models']['cnn_input_shape'], config['models']['num_classes'], config['training']['dropout'])
    else:
        model = GenreLSTM(
            input_size=config['data']['n_mfcc'],
            hidden_size=128,
            num_layers=2,
            num_classes=config['models']['num_classes'],
            dropout=config['training']['dropout']
        )
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"results/checkpoints/best_{model_type}.pth")
    
    print("Training complete.")
