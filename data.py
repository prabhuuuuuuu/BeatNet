import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import LabelEncoder #ganre strings to integers
import torch
from torch.utils.data import Dataset, DataLoader
from features import extract_mfcc, extract_mel_spectrogram

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class GenreDataset(Dataset):
    def __init__(self, file_list, labels, model_type = 'cnn', config = config['data']):
        self.file_list = file_list
        self.labels = labels
        self.model_type = model_type
        self.config = config

    def __len__(self):
        return len(self.file_list) #number of filess, basically returns dataset size
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        if self.model_type == 'cnn':
            features = extract_mel_spectrogram(audio_path, self.config['sample_rate'], self.config['n_mels'],
                                                self.config['hop_length'], self.config['n_fft'], self.config['duration'])
        
        else: #if no CNN, its LSTM for MFCC
            features = extract_mfcc(audio_path, self.config['sample_rate'], self.config['n_mfcc'], self.config['hop_length'], self.config['n_fft'], self.config['duration'])

        features = torch.tensors(features - np.mean(features)) / np.std(features) #standardize features for stable training

        features = torch.tensor(features, dtype = torch.float32) #numpy to tensor
        label = torch.tensor(label, dtype = torch.long) #label to tensor

        return features, label
    
def load_and_split_data(dataset_path, config):
    file_list = []
    genre_list = []

    genres = os.listdir(dataset_path)

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in os.listdr(genre_path):
            if file.endswith('.wav'):
                file_list.append(os.path.join(genre_path, file))
                genre_list.append(genre)

    
    label_encoder = LabelEncoder()

    labels = label_encoder.fit_trasnform(genre_list)

    train_files, temp_files, train_labels, temp_labels = train_test_split(file_list, labels, train_size = config['split']['train_size'], stratify=labels, random_state=config['states']['random_seed'])

    val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, train_size = config['split']['val_size'], stratify=temp_labels, random_state=config['states']['random_seed'])

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

def get_dataloaders(model_type = 'cnn'): #to create dataloader
    train_data, val_data, test_data, label_encoder = load_and_split_data(config['data']['dataset_path'], config)
    train_dataset = GenreDataset(train_data[0], train_data[1], model_type)
    val_dataset = GenreDataset(val_data[0], val_data[1], model_type)
    test_dataset = GenreDataset(test_data[0], test_data[1], model_type)

    train_loader = DataLoader(train_dataset, batch_size = config['training']['batch_size'], shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = config['training']['batch_size'], shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = config['training']['batch_size'], shuffle = False)

    return train_loader, val_loader, test_loader, label_encoder







