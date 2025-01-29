import numpy as np
import torch
from librosa import power_to_db, feature
from librosa.feature import melspectrogram
import soundfile as sf
import utils.Data_Transforms_for_SER as ser


def extract_features(audio_data, fs, frame_l=2**10, hop_l=2**8):
    """
    Extracts MFCC features from audio signals.

    Args:
        audio_data (list): List of raw audio signals.
        fs (int): Sampling rate.

    Returns:
        torch.Tensor: Preprocessed dataset as a tensor.
    """
    sgram = []
    L_sgram = []

    for signal in audio_data:
        mel = melspectrogram(y=signal.astype(np.float32), sr=fs, n_fft=frame_l, hop_length=hop_l)
        mel_dB = power_to_db(mel, ref=np.max)
        sgram.append(mel_dB)
        L_sgram.append(len(mel[0, :]))

    L_q_sgram_max = 20  # Fixed size for quadratic transformation
    q_sgram=[]

    for i in range(len(L_sgram)):
        new_mel=ser.resize_transform(sgram[i],L_q_sgram_max,L_q_sgram_max)
        q_sgram.append(new_mel)

    return torch.FloatTensor(q_sgram).reshape(len(q_sgram), 1, L_q_sgram_max, L_q_sgram_max)

def resize_transform(spectrogram, new_height, new_width):
    """
    Resizes a spectrogram to a fixed shape.
    """
    return np.resize(spectrogram, (new_height, new_width))
