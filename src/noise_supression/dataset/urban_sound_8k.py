from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.noise_supression.dataset._const import NP_RANDOM_SEED

np.random.seed(NP_RANDOM_SEED)


class UrbanSound8k:
    def __init__(self, basepath: Path, val_dataset_size: int, class_ids: np.ndarray = None) -> None:
        self.__basepath: Path = basepath
        self.__val_dataset_size: int = val_dataset_size
        self.__class_ids: np.ndarray = class_ids
        self.__metadata: pd.DataFrame = self.__get_metadata()

    def __get_metadata(self) -> pd.DataFrame:
        print('Getting U8K metadata...')

        metadata: pd.DataFrame = pd.read_csv(self.__basepath / 'UrbanSound8K.csv')
        metadata.reindex(np.random.permutation(metadata.index))

        return metadata

    def __get_filenames_by_class_id(self, metadata: pd.DataFrame) -> List[str]:
        if self.__class_ids is None:
            self.__class_ids = np.unique(self.__metadata['classID'].values)

        files: List[str] = []

        for class_id in self.__class_ids:
            per_id_files: np.ndarray = metadata[metadata['classID'] == class_id][['slice_file_name', 'fold']].values
            files.extend([str(self.__basepath / ('fold' + str(file[1])) / file[0]) for file in per_id_files])

        return files

    def get_train_val_filenames(self) -> Tuple[List[str], List[str]]:
        train_meta: pd.DataFrame = self.__metadata[self.__metadata.fold != 10]

        filenames: List[str] = self.__get_filenames_by_class_id(train_meta)
        np.random.shuffle(filenames)

        val: List[str] = filenames[-self.__val_dataset_size:]
        train: List[str] = filenames[:-self.__val_dataset_size]

        print(f'Train samples: {len(train)} | Val samples: {len(val)}')

        return train, val

    def get_test_filenames(self) -> List[str]:
        test_meta: pd.DataFrame = self.__metadata[self.__metadata.fold == 10]

        test: List[str] = self.__get_filenames_by_class_id(test_meta)
        np.random.shuffle(test)

        print(f'Test samples: {len(test)}')

        return test
