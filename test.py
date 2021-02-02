from pathlib import Path

import soundfile as sf

from src.noise_supression import suppress_noise_without_sample

signal = suppress_noise_without_sample(Path(r''),
                                       Path(r''),
                                       16000)

sf.write(r'', signal, 16000, format='WAV')
