from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.signal import convolve

from src.noise_suppression.nn._demucs import Demucs

_DEMUCS_CFG = {
    'chin': 1,
    'chout': 1,
    'hidden': 48,
    'max_hidden': 10000,
    'causal': True,
    'glu': True,
    'depth': 5,
    'kernel_size': 8,
    'stride': 4,
    'normalize': True,
    'resample': 4,
    'growth': 2,
    'rescale': 0.1,
}


class NeuralNetworkNoiseSuppressor:
    def __init__(self, weights_path: Path) -> None:
        checkpoint = torch.load(weights_path)
        self.__model = Demucs(**_DEMUCS_CFG)
        self.__model.load_state_dict(checkpoint)

        self.__filter = [0.5, 0.75, 1, 0.75, 0.5]

    def suppress(self, audio_path: Path, sample_rate: int, device: str = 'cpu'):
        signal, sr = librosa.load(audio_path, sample_rate)
        signal = convolve(signal, self.__filter, mode='same')
        signal /= np.max(np.abs(signal))

        signal_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        if device == 'cuda':
            signal_torch = signal_torch.to(device)
            self.__model.to(device)

        signal = self.__enhance(signal_torch.unsqueeze(0), device).numpy()
        signal /= np.max(np.abs(signal))

        return signal, sr

    def __enhance(self, noisy_mix, device: str, sample_len: int = 16384):
        padded_length = 0

        if noisy_mix.size(-1) % sample_len != 0:
            padded_length = sample_len - (noisy_mix.size(-1) % sample_len)
            noisy_mix = torch.cat(
                [noisy_mix, torch.zeros(size=(1, 1, padded_length), device=device)], dim=-1
            )

        assert noisy_mix.size(-1) % sample_len == 0 and noisy_mix.dim() == 3

        noisy_chunks = list(torch.split(noisy_mix, sample_len, dim=-1))
        noisy_chunks = torch.cat(noisy_chunks, dim=0)

        enhanced_chunks = self.__model(noisy_chunks).detach().cpu()

        enhanced = enhanced_chunks.reshape(-1)

        if padded_length != 0:
            enhanced = enhanced[:-padded_length]
            noisy_mix = noisy_mix[:-padded_length]

        return enhanced
