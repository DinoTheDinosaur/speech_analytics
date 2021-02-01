import librosa
import numpy as np


def read_audio(path, sample_rate, normalize=True):
    audio, sr = librosa.load(path, sample_rate)

    if normalize:
        librosa.util.normalize(audio)

    return audio, sr


def add_noise(clean_audio, noise):
    while clean_audio.size >= noise.size:
        noise = np.append(noise, noise)

    i = np.random.randint(0, noise.size - clean_audio.size)
    noise_segment = noise[i:i + clean_audio.size]

    clean_power = np.sum(np.square(clean_audio))
    noise_power = np.sum(np.square(noise_segment))

    return clean_audio + np.sqrt(clean_power / noise_power) * noise_segment

# def _signal_rms(signal: np.ndarray) -> float:
#     return np.sqrt(np.sqrt(np.mean(np.sqr(signal))))
#
#
# def _noise_rms(sig_rms: float, target_snr: float = 20) -> float:
#     return np.sqrt(sig_rms * sig_rms / np.power(10, target_snr / 10))
#
#
# def add_awgn(signal: np.ndarray, target_snr: float = 20) -> np.ndarray:
#     sig_rms = _signal_rms(signal)
#     noise_rms = _noise_rms(sig_rms, target_snr)
#
#     noise = np.random.normal(0, noise_rms, signal.shape[0])
#
#     return signal + noise
#
#
# def add_real_world_noise(signal: np.ndarray, noise: np.ndarray, target_snr: float = 20) -> np.ndarray:
#     sig_rms = _signal_rms(signal)
#     cur_noise_rms = _signal_rms(noise)
#     target_noise_rms = _noise_rms(sig_rms, target_snr)
#
#     coef = target_noise_rms / cur_noise_rms
#
#     return signal + coef * noise
