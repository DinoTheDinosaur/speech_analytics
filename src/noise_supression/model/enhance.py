from pathlib import Path

import librosa
import torch

from src.noise_supression.model._demucs import Demucs

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


def _enhance(model, noisy_mix, device, sample_len=16384):
    padded_length = 0

    if noisy_mix.size(-1) % sample_len != 0:
        padded_length = sample_len - (noisy_mix.size(-1) % sample_len)
        noisy_mix = torch.cat(
            [noisy_mix, torch.zeros(size=(1, 1, padded_length), device=device)], dim=-1
        )

    assert noisy_mix.size(-1) % sample_len == 0 and noisy_mix.dim() == 3

    noisy_chunks = list(torch.split(noisy_mix, sample_len, dim=-1))
    noisy_chunks = torch.cat(noisy_chunks, dim=0)

    enhanced_chunks = model(noisy_chunks).detach().cpu()

    enhanced = enhanced_chunks.reshape(-1)

    if padded_length != 0:
        enhanced = enhanced[:-padded_length]
        noisy_mix = noisy_mix[:-padded_length]

    return enhanced


def suppress_noise_without_sample(model_weights_path: Path, audio_path: Path, sample_rate: int, device: str = 'cpu'):
    checkpoint = torch.load(model_weights_path)
    model = Demucs(**_DEMUCS_CFG)
    model.load_state_dict(checkpoint)

    signal, _ = librosa.load(audio_path, sample_rate)
    signal_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    if device == 'cuda':
        signal_torch = signal_torch.to(device)
        model.to(device)

    return _enhance(model, signal_torch.unsqueeze(0), device)
