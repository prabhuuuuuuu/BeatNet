#defines our CNN and LSTM models

import torch.nn as n
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class GenreCNN(nn.Module):
    #we will use CNN for mel spectrograms

    def __init__(self, input_shape, num_classes, dropout):
        super(GenreCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) #flatten
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    


class GenreLSTM(nn.Module):
    #LSTM for MFCC sequences

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(GenreLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional = True) #bidirectional catches dependancies in both directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x) #ignores output, takes last h_n
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim = 1) #torch cat contatenates tensors
        x = self.dropout(h_n)
        x = self.fc(x)
        return x
n

