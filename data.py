import os
import yaml
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import librosa  # For loading and augmentations.
from features import extract_mfcc, extract_mel_spectrogram

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class GenreDataset(Dataset):
    def __init__(self, file_list, labels, model_type='cnn', config=config['data'], is_train=False):
        self.file_list = file_list
        self.labels = labels
        self.model_type = model_type
        self.config = config
        self.is_train = is_train
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'], duration=self.config['duration'])
        except Exception as e:
            print(f"Error loading {audio_path}: {e}. Using dummy.")
            audio = np.zeros(int(self.config['duration'] * self.config['sample_rate']))
            sr = self.config['sample_rate']
        
        # Apply random augmentations if training (50% chance)
        if self.is_train and random.random() < 0.5:
            # Randomly select and apply one augmentation
            aug_choice = random.choice(['noise', 'shift', 'pitch', 'stretch'])
            if aug_choice == 'noise':
                noise = np.random.randn(len(audio))
                audio = audio + 0.005 * noise
            elif aug_choice == 'shift':
                shift = int(sr * 0.2 * (2 * random.random() - 1))
                audio = np.roll(audio, shift)
            elif aug_choice == 'pitch':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-2, 2))
            elif aug_choice == 'stretch':
                rate = random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)
                # Adjust length back to original if stretched
                target_len = int(self.config['duration'] * sr)
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
                else:
                    audio = audio[:target_len]
        
        # Extract features from augmented audio
        if self.model_type == 'cnn':
            features = extract_mel_spectrogram(audio, sr, self.config['n_mels'],
                                               self.config['hop_length'], self.config['n_fft'], 130)
        else:
            features = extract_mfcc(audio, sr, self.config['n_mfcc'],
                                    self.config['hop_length'], self.config['n_fft'], 1291)
        
        if features is None:
            print(f"Using dummy features for {audio_path}.")
            if self.model_type == 'cnn':
                features = np.zeros((1, self.config['n_mels'], 130))
            else:
                features = np.zeros((1291, self.config['n_mfcc']))
        
        # Safe normalization
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
        file_list, labels, train_size=config['splits']['train_size'], stratify=labels, 
        random_state=config['splits']['random_seed'])
    remaining_size = config['splits']['val_size'] + config['splits']['test_size']
    val_ratio = config['splits']['val_size'] / remaining_size
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, train_size=val_ratio, stratify=temp_labels, 
        random_state=config['splits']['random_seed'])
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), label_encoder

def get_dataloaders(model_type='cnn'):
    train_data, val_data, test_data, label_encoder = load_and_split_data(config['data']['dataset_path'], config)
    train_dataset = GenreDataset(train_data[0], train_data[1], model_type, is_train=True)
    val_dataset = GenreDataset(val_data[0], val_data[1], model_type, is_train=False)
    test_dataset = GenreDataset(test_data[0], test_data[1], model_type, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader, label_encoder
