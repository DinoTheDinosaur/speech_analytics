import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from src.utils import validate_dir
from src.noise_supression.dataset.dataset import Dataset
from src.noise_supression.dataset.mcv import MCV
from src.noise_supression.dataset.urban_sound_8k import UrbanSound8k

EXIT_CODE: int = 1
WINDOW_LEN: int = 256

DEFAULTS = {
    'window_len': WINDOW_LEN,
    'overlap': round(0.25 * WINDOW_LEN),
    'sample_rate': 16000,
    'audio_max_duration': 0.8
}


def create_set(out_path: Path, mcv_filenames: List[str], u8k_filenames: List[str], config=None) -> None:
    if not validate_dir(out_path):
        out_path.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULTS if config is None else config

    ds = Dataset(out_path, **cfg)
    ds.create(mcv_filenames, u8k_filenames)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mcv_dir', type=str, help='Directory with MCV dataset.')
    parser.add_argument('urban8k_dir', type=str, help='Directory with UrbanSound8K dataset.')
    parser.add_argument('out_dir', type=str, help='Directory for resulting files.')

    args = parser.parse_args()
    mcv_dir = Path(args.mcv_dir)
    urban8k_dir = Path(args.urban8k_dir)
    out_dir = Path(args.out_dir)

    if not validate_dir(mcv_dir):
        sys.stderr.write('Path to MCV doesn\'t exist or isn\'t a directory')
        sys.exit(EXIT_CODE)

    if not validate_dir(urban8k_dir):
        sys.stderr.write('Path to UrbanSound8K doesn\'t exist or isn\'t a directory')
        sys.exit(EXIT_CODE)

    if not validate_dir(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    print('Getting MCV train, val and test filenames...')
    mcv = MCV(mcv_dir, 1000)
    mcv_train_filenames, mcv_val_filenames = mcv.get_train_val_filenames()
    mcv_test_filenames = mcv.get_test_filenames()

    print('Getting U8K train, val and test filenames...')
    u8k = UrbanSound8k(urban8k_dir, 200)
    u8k_train_filenames, u8k_val_filenames = u8k.get_train_val_filenames()
    u8k_test_filenames = u8k.get_test_filenames()

    train_out_dir = out_dir / 'train'

    print('Applying noise to train data...')
    create_set(train_out_dir, mcv_train_filenames, u8k_train_filenames)

    val_out_dir = out_dir / 'val'

    print('Applying noise to val data...')
    create_set(val_out_dir, mcv_val_filenames, u8k_val_filenames)

    test_out_dir = out_dir / 'test'

    print('Applying noise to test data...')
    create_set(test_out_dir, mcv_test_filenames, u8k_test_filenames)

    print('DONE')
