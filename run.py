from pathlib import Path
from pprint import pprint

import yaml

from src.processing.audio_processor import AudioProcessor

CONFIG_PATH: Path = Path(__file__).parent / 'config.yaml'

if __name__ == '__main__':
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ap = AudioProcessor(Path(cfg['suppressor_model_weights']),
                        Path(cfg['vosk_model']),
                        Path(cfg['white_list']),
                        Path(cfg['obscene_corpus']),
                        Path(cfg['threats_corpus']),
                        Path(cfg['white_checklist']),
                        Path(cfg['black_checklist']))
    out = ap.process(Path(cfg['audio_path']))

    print('OPERATOR EVALUATION')
    pprint(out)
