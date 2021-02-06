import logging
from pathlib import Path

import yaml

from src.bot import SpeechAnalyticsBot

CONFIG_PATH: Path = Path(__file__).parent / 'config.yaml'
DATA_PATH: Path = Path(__file__).parent / 'data'

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    token = cfg['bot_token']
    del cfg['bot_token']

    bot = SpeechAnalyticsBot(token, cfg, DATA_PATH)
    bot.run()
