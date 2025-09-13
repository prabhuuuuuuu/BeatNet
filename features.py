# features.py: Extracts features from raw audio arrays with guaranteed fixed dimensions.
import librosa
import numpy as np
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_mfcc(audio, sr, n_mfcc, hop_length, n_fft, target_seq_len=1291):
    """
    Extracts MFCCs and resizes them to a fixed sequence length.
    """
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mfcc = mfcc.T  # Transpose to [time, features]

        # Pad or truncate to the target sequence length
        if mfcc.shape[0] < target_seq_len:
            pad_len = target_seq_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        elif mfcc.shape[0] > target_seq_len:
            mfcc = mfcc[:target_seq_len, :]
            
        return mfcc
    except Exception as e:
        print(f"Error extracting MFCC: {e}. Returning zeros.")
        return np.zeros((target_seq_len, n_mfcc))

def extract_mel_spectrogram(audio, sr, n_mels, hop_length, n_fft, target_width=130):
    """
    Extracts a Mel spectrogram and resizes it to a fixed width.
    """
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or truncate the width (time dimension) to the target width
        if mel_spec_db.shape[1] < target_width:
            pad_width = target_width - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), 'constant', constant_values=mel_spec_db.min())
        elif mel_spec_db.shape[1] > target_width:
            mel_spec_db = mel_spec_db[:, :target_width]
            
        # Add channel dimension for CNN
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        return mel_spec_db
    except Exception as e:
        print(f"Error extracting Mel spectrogram: {e}. Returning zeros.")
        return np.zeros((1, n_mels, target_width))

