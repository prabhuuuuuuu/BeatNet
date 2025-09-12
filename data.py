import os
import yaml
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from features import extract_mfcc, extract_mel_spectrogram
import librosa

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- Augmentation Functions ---
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, sr, shift_max_sec=0.2):
    shift = int(sr * shift_max_sec * (2 * random.random() - 1))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps=4):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps * (2 * random.random() - 1))

def time_stretch(y, rate_min=0.8, rate_max=1.2):
    rate = random.uniform(rate_min, rate_max)
    return librosa.effects.time_stretch(y=y, rate=rate)


class GenreDataset(Dataset):
    def __init__(self, file_list, labels, model_type='cnn', is_train=False, config_data=config['data']):
        self.file_list = file_list
        self.labels = labels
        self.model_type = model_type
        self.config = config_data
        self.is_train = is_train
        self.augmentations = [add_noise, time_shift, pitch_shift, time_stretch]

    def __len__(self):
        return len(self.file_list)

    def apply_augmentations(self, y, sr):
        # Apply one random augmentation
        aug = random.choice(self.augmentations)
        if aug.__name__ in ['time_shift', 'pitch_shift']:
            y = aug(y, sr)
        else:
            y = aug(y)
        return y

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        try:
            audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'], duration=self.config['duration'])
        except Exception as e:
            print(f"Error loading {audio_path}: {e}. Using silent audio.")
            audio = np.zeros(self.config['sample_rate'] * self.config['duration'])
            sr = self.config['sample_rate']

        # Apply augmentations to 50% of training samples
        if self.is_train and random.random() < 0.5:
            audio = self.apply_augmentations(audio, sr)
        
        if self.model_type == 'cnn':
            features = extract_mel_spectrogram(
                audio, sr,
                self.config['n_mels'],
                self.config['hop_length'],
                self.config['n_fft']
            )
        else:
            features = extract_mfcc(
                audio, sr,
                self.config['n_mfcc'],
                self.config['hop_length'],
                self.config['n_fft']
            )
        
        if features is None:
            print(f"Using dummy features for {audio_path}.")
            if self.model_type == 'cnn':
                features = np.zeros((1, self.config['n_mels'], 130))
            else:
                features = np.zeros((1291, self.config['n_mfcc']))
        
        mean = np.mean(features)
        std = np.std(features)
        if std == 0:
            features = features - mean
        else:
            features = (features - mean) / std
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

def load_and_split_data(dataset_path, config):
    file_list = []
    genre_list = []
    
    items = os.listdir(dataset_path)
    genres = [item for item in items if os.path.isdir(os.path.join(dataset_path, item))]
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                full_path = os.path.join(genre_path, file)
                try:
                    librosa.load(full_path, sr=config['data']['sample_rate'], duration=1)
                    file_list.append(full_path)
                    genre_list.append(genre)
                except Exception as e:
                    print(f"Skipping corrupt file: {full_path} ({e})")
    
    if not file_list:
        raise ValueError("No valid audio files found! Check dataset.")
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(genre_list)
    
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels,
        train_size=config['splits']['train_size'],
        stratify=labels,
        random_state=config['splits']['random_seed']
    )
    
    remaining_size = config['splits']['val_size'] + config['splits']['test_size']
    val_ratio = config['splits']['val_size'] / remaining_size
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=config['splits']['random_seed']
    )
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), label_encoder

def get_dataloaders(model_type='cnn'):
    train_data, val_data, test_data, label_encoder = load_and_split_data(config['data']['dataset_path'], config)
    
    # Pass is_train=True for the training set to enable augmentations
    train_dataset = GenreDataset(train_data[0], train_data[1], model_type, is_train=True)
    val_dataset = GenreDataset(val_data[0], val_data[1], model_type, is_train=False)
    test_dataset = GenreDataset(test_data[0], test_data[1], model_type, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader, label_encoder