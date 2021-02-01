import sys
from argparse import ArgumentParser
from pathlib import Path

from src.noise_supression._utils import validate_dir
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

    # get filenames of train and validation datasets
    print('Getting MCV train and val filenames...')
    mcv = MCV(mcv_dir, 1000)
    mcv_train_filenames, mcv_val_filenames = mcv.get_train_val_filenames()

    print('Getting U8K train and val filenames...')
    u8k = UrbanSound8k(urban8k_dir, 200)
    u8k_train_filenames, u8k_val_filenames = u8k.get_train_val_filenames()

    # create train dataset
    train_out_dir = out_dir / 'train'

    if not validate_dir(train_out_dir):
        train_out_dir.mkdir(parents=True, exist_ok=True)

    print('Applying noise to train data...')
    train = Dataset(train_out_dir, **DEFAULTS)
    train.create(mcv_train_filenames[:100], u8k_train_filenames)

    # create validation dataset
    # val_out_dir = out_dir / 'train'
    #
    # if not validate_dir(val_out_dir):
    #     val_out_dir.mkdir(parents=True, exist_ok=True)
    #
    # print('Applying noise to val data...')
    # val = Dataset(val_out_dir, **DEFAULTS)
    # val.create(mcv_val_filenames, u8k_val_filenames)

    print('DONE')
    # create test dataset
