import os
import torch

import hydra
import librosa
import soundfile as sf

from demucs import Demucs


def enhance(model, noisy_mix, sample_len):
    if noisy_mix.size(-1) % sample_len != 0:
        padded_length = sample_len - (noisy_mix.size(-1) % sample_len)
        noisy_mix = torch.cat(
            [noisy_mix, torch.zeros(size=(1, 1, padded_length))], dim=-1
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


@hydra.main(config_name="inference_conf.yaml")
def cleaner(cfg):
    checkpoint = torch.load(cfg.model_weights, map_location=cfg.device)
    model = Demucs(**cfg.demucs)
    model.load_state_dict(checkpoint)
    signal, sr = librosa.load(cfg.file, cfg.sr)
    signal_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    if cfg.device == "gpu":
        signal_torch.to("gpu")

    enhanced_signal = enhance(model, signal_torch.unsqueeze(0), cfg.sample_len)
    os.makedirs(cfg.save_dir, exist_ok=True)
    f_name = cfg.file.split("/")[-1].replace(".wav", "_clean.wav")
    save_path = os.path.join(cfg.save_dir, f_name)
    sf.write(save_path, enhanced_signal, cfg.sr)


if __name__ == "__main__":
    cleaner()