import torch
import yaml
from models import GenreCNN, GenreLSTM
from features import extract_mfcc, extract_mel_spectrogram
import numpy as np
import librosa

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def infer_genre(custom_audio_path, model_type='cnn', checkpoint_path=None, label_encoder=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load audio first
    audio, sr = librosa.load(
        custom_audio_path, 
        sr=config['data']['sample_rate'], 
        duration=config['data']['duration']
    )
    
    if model_type == 'cnn':
        model = GenreCNN(config['models']['cnn_input_shape'], config['models']['num_classes'], config['training']['dropout'])
        features = extract_mel_spectrogram(
            audio, sr,
            config['data']['n_mels'],
            config['data']['hop_length'],
            config['data']['n_fft']
        )
    else:
        model = GenreLSTM(
            input_size=config['data']['n_mfcc'],
            hidden_size=128,
            num_layers=2,
            num_classes=config['models']['num_classes'],
            dropout=config['training']['dropout']
        )
        features = extract_mfcc(
            audio, sr,
            config['data']['n_mfcc'],
            config['data']['hop_length'],
            config['data']['n_fft']
        )
    
    features = (features - np.mean(features)) / np.std(features)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(features)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    genre = label_encoder.inverse_transform([pred])[0]
    
    print(f"Predicted Genre: {genre}")
    return genre