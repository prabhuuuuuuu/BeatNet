import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import GenreCNN, GenreLSTM
from data import get_dataloaders
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def evaluate_model(model_type='cnn', checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, test_loader, label_encoder = get_dataloaders(model_type)
    
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
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({model_type.upper()})')
    plt.savefig(f'results/plots/conf_matrix_{model_type}.png')
    
    print(f"Accuracy: {accuracy:.4f}")
    for i, genre in enumerate(label_encoder.classes_):
        print(f"{genre}: Precision {precision[i]:.4f}, Recall {recall[i]:.4f}, F1 {f1[i]:.4f}")
    
    return accuracy, conf_matrix
