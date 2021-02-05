import logging
from pathlib import Path

import ffmpeg
import yaml
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram.ext import Updater

from src.processing.audio_processor import AudioProcessor

CONFIG_PATH: Path = Path(__file__).parent / 'config.yaml'
DATA_PATH: Path = Path(__file__).parent / 'data'
IN_PATH: Path = DATA_PATH / 'file.wav'
OUT_PATH: Path = DATA_PATH / 'file_proc.wav'

GREETING: str = "Hi, I'm CallCenterAnalyticsBot. Ready to get dialog recording (in .wav) and return a report about " \
                "operator's work."

AUDIO_PROC = None
REPORT: str = ''


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=GREETING)


def wav_downloader(update, context):
    global REPORT

    context.bot.get_file(update.message.document).download()
    context.bot.send_message(chat_id=update.effective_chat.id, text="File for analysis received")

    with open(DATA_PATH / "file.wav", 'wb') as f:
        context.bot.get_file(update.message.document).download(out=f)

    context.bot.send_message(chat_id=update.effective_chat.id, text="File saved")
    context.bot.send_message(chat_id=update.effective_chat.id, text="Starting processing")
    REPORT = str(process_file())
    print_report(update, context)


def print_report(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=REPORT)


def process_file():

    out, _ = (ffmpeg.
              input(str(IN_PATH.resolve())).
              output(str(OUT_PATH.resolve()), acodec='pcm_s16le', ac=1, ar='8000').
              overwrite_output().
              run()
              )

    IN_PATH.unlink()
    OUT_PATH.unlink()

    return AUDIO_PROC.process(DATA_PATH / 'file_proc.wav')


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print('initializing audio processor...')
    AUDIO_PROC = AudioProcessor(Path(cfg['suppressor_model_weights']),
                                Path(cfg['vosk_model']),
                                Path(cfg['white_list']),
                                Path(cfg['obscene_corpus']),
                                Path(cfg['threats_corpus']),
                                Path(cfg['white_checklist']),
                                Path(cfg['black_checklist']),
                                cfg['recognition_engine'],
                                cfg['bucket'],
                                cfg['aws_key'],
                                cfg['aws_key_id'],
                                cfg['ya_api_key'])

    updater = Updater(token=cfg['bot_token'], use_context=True)
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)

    dispatcher.add_handler(echo_handler)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(MessageHandler(Filters.document.category('audio/'), wav_downloader))

    print('starting bot...')
    updater.start_polling()
    updater.idle()
