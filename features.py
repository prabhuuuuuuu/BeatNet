import librosa
import numpy as np

def extract_mfcc(audio_path, sample_rate, n_mfcc, hop_length, n_fft, duration):
    audio, sr = librosa.load(audio_path, sr = sample_rate, duration = duration)
    mfcc = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = n_mfcc, hop_length = hop_length, n_fft = n_fft)
    mfcc = np.mean(mfcc, axis = 1) if mfcc.shape[1] == 0 else mfcc
    return mfcc.T # .T for transpose for LSTM

def extract_mel_spectrogram(audio_path, sample_rate, n_mels, hop_length, n_fft, duration):
    audio, sr = librosa.load(audio_path, sr = sample_rate, duration = duration)
    mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft)
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
    mel_spec_db = np.expand_dims(mel_spec_db, axis = 0) #to make it CNN compatible 
    return mel_spec_db #returns as a numpy array


'''
this block extracts key audio features like MFCC for sequence and Mel spectrogram for images. it transforms raw sound into usable representations for ML models.
'''