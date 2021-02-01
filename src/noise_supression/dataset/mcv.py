from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.noise_supression.dataset._const import NP_RANDOM_SEED

np.random.seed(NP_RANDOM_SEED)


class MCV:
    def __init__(self, basepath: Path, val_dataset_size: int) -> None:
        self.__basepath: Path = basepath
        self.__val_dataset_size: int = val_dataset_size

    def __get_filenames(self, df_name: str) -> np.ndarray:
        print('Getting MCV metadata...')

        metadata: pd.DataFrame = pd.read_csv(self.__basepath / df_name, sep='\t')
        
        files: np.ndarray = metadata['path'].values
        np.random.shuffle(files)

        return files

    def get_train_val_filenames(self) -> Tuple[List[str], List[str]]:
        files = [str(self.__basepath / 'clips' / filename) for filename in self.__get_filenames('train.tsv')]

        train: List[str] = files[:-self.__val_dataset_size]
        val: List[str] = files[-self.__val_dataset_size:]

        print(f'Train samples: {len(train)} | Val samples: {len(val)}')

        return train, val

    def get_test_filenames(self) -> List[str]:
        test = [str(self.__basepath / 'clips' / filename) for filename in self.__get_filenames('test.tsv')]

        print(f'Test samples: {len(test)}')

        return test
