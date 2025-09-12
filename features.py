import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_mfcc(audio_path, sample_rate, n_mfcc, hop_length, n_fft, duration, target_seq_len=1291):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mfcc = mfcc.T
        
        if mfcc.shape[0] < target_seq_len:
            pad_len = target_seq_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        elif mfcc.shape[0] > target_seq_len:
            mfcc = mfcc[:target_seq_len, :]
        
        return mfcc
    except Exception as e:
        print(f"Error extracting MFCC from {audio_path}: {e}. Skipping file.")
        return None

def extract_mel_spectrogram(audio_path, sample_rate, n_mels, hop_length, n_fft, duration, target_width=130):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] < target_width:
            pad_width = target_width - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), 'constant', constant_values=mel_spec_db.min())
        elif mel_spec_db.shape[1] > target_width:
            mel_spec_db = mel_spec_db[:, :target_width]
        
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        return mel_spec_db
    except Exception as e:
        print(f"Error extracting Mel spectrogram from {audio_path}: {e}. Skipping file.")
        return None
