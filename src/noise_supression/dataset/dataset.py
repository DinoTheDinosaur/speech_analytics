import multiprocessing as mp
import warnings
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from src.noise_supression.dataset._utils import add_noise, read_audio

warnings.filterwarnings('ignore', category=UserWarning)

_CORES: int = mp.cpu_count()


class Dataset:
    def __init__(self, out_path: Path, **kwargs) -> None:
        self.__out_path: Path = out_path
        self.__sample_rate: int = kwargs['sample_rate']
        self.__overlap: int = kwargs['overlap']
        self.__window_length: int = kwargs['window_len']
        self.__audio_max_duration: float = kwargs['audio_max_duration']

    def process_clean_file(self, clean_file, noise):
        clean_audio, _ = read_audio(clean_file, self.__sample_rate)
        noised = add_noise(clean_audio, noise)

        sf.write(self.__out_path / Path(clean_file).name, noised, self.__sample_rate, format='WAV')

    def create(self, clean_filenames: List[str], noise_filenames: List[str]) -> None:
        print('Creating arguments list for Pool...')
        args = ((clean_file, read_audio(np.random.choice(noise_filenames), self.__sample_rate)[0]) for clean_file in
                clean_filenames)

        print('Starting processing...')
        q = mp.Queue(maxsize=_CORES)

        with mp.Pool(_CORES) as pool:
            pool.starmap(self.process_clean_file, args)

        # for clean_file in clean_filenames:
        #     clean_audio, _ = read_audio(clean_file, self.__sample_rate)
        #     noise, _ = read_audio(np.random.choice(noise_filenames), self.__sample_rate)
        #
        #     noised = add_noise(clean_audio, noise)
        #
        #     sf.write(out_path / Path(clean_file).name, noised, self.__sample_rate, format='WAV')
