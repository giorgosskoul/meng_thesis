import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spectrogram):
    """
    Plots a spectrogram.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()
