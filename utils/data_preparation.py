import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time  # If execution time needs to be measured
import matplotlib.pylab as plt
import pandas as pd
import random
from gc import collect
from PIL import Image
import os
from os.path import join, isfile
from scipy.io import wavfile
from librosa.effects import preemphasis
# Import transformations from the SER module
import utils.Data_Transforms_for_SER as ser




def load_data(directory):
    """
    Loads WAV files from all subdirectories in the specified dataset directory 
    and assigns emotion labels.

    Args:
        directory (str): Path to the dataset folder.

    Returns:
        tuple: A list of raw audio signals and their corresponding labels.
    """
    audio_data = []
    labels = []

    # Traverse all subdirectories in the dataset folder
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Ensure it's a WAV file
                file_path = join(root, file)
                fs, signal = wavfile.read(file_path)

                # Handle stereo files by keeping only one channel
                if signal.ndim > 1:
                    signal = signal[:, 0]

                audio_data.append(signal)

                # Assign labels based on filename structure
                if file.startswith('03-01-01'):
                    labels.append("Neutral")
                elif file.startswith('03-01-02'):
                    labels.append("Calm")
                elif file.startswith('03-01-03'):
                    labels.append("Happy")
                elif file.startswith('03-01-04'):
                    labels.append("Sad")
                elif file.startswith('03-01-05'):
                    labels.append("Angry")
                elif file.startswith('03-01-06'):
                    labels.append("Fearful")
                elif file.startswith('03-01-07'):
                    labels.append("Disgust")
                elif file.startswith('03-01-08'):
                    labels.append("Surprised")
                else:
                    labels.append("error")

    return audio_data, labels



def preprocess_data(audio_data, labels):
    """
    Applies preprocessing to the audio data (trimming zeros and pre-emphasis).

    Args:
        audio_data (list): List of raw audio signals.
        labels (list): Corresponding emotion labels.

    Returns:
        list: Processed audio signals.
        list: Corresponding labels.
    """
    processed_data = []
    processed_labels = []

    for i in range(len(audio_data)):
        processed_signal = np.trim_zeros(
            np.trim_zeros(preemphasis(audio_data[i].astype(np.float32)), 'f'), 'b'
        )
        processed_data.append(processed_signal)
        processed_labels.append(labels[i])

    return processed_data, processed_labels

