from pathlib import Path
from typing import Dict

import audioread
import numpy as np
import scipy.signal
from scipy.io import wavfile
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater, Dispatcher

from src.processing import AudioProcessor

DEFAULT_GREETING: str = "Hi, I'm CallCenterAnalyticsBot. Ready to get dialog recording (in .wav) and return a report " \
                        "about operator's work."


def _audio_decoder(path2file, sample_rate=8000, write_path=None):
    """
    Input:
        path2file - название файла и путь до него
        sample_rate - желаемая на выходе частота дискертизации (по умолчанию 8кГц)
        write_path - путь и название сохраняемого файла, по умолчанию "None"
    Output:
        np.array()
    """
    audio = np.array([])

    with audioread.audio_open(path2file) as bytes_file:
        n_channels = bytes_file.channels
        sr = bytes_file.samplerate
        duration = bytes_file.duration

        for buf in bytes_file:
            part = np.frombuffer(buf, dtype=np.int16)
            audio = np.concatenate((audio, part))

    audio = audio / abs(audio).max()

    if sr != sample_rate:
        new_samps = int(duration * sample_rate)
        audio = scipy.signal.resample(audio, new_samps)

    if write_path:
        wavfile.write(write_path, rate=sample_rate, data=audio)

    return audio


class SpeechAnalyticsBot:
    def __init__(self, token: str, processor_config: Dict, data_path: Path, greeting: str = DEFAULT_GREETING) -> None:
        self.__updater = Updater(token=token, use_context=True)
        self.__dispatcher: Dispatcher = self.__updater.dispatcher
        self.__processor = AudioProcessor(**processor_config)

        self.__data_path: Path = data_path
        self.__in_path: Path = self.__data_path / 'file.wav'
        self.__out_path: Path = self.__data_path / 'file_proc.wav'

        self.__report: str = ''
        self.__greeting: str = greeting

        start_handler = CommandHandler('start', self.__start)
        echo_handler = MessageHandler(Filters.text & (~Filters.command), self.__echo)

        self.__dispatcher.add_handler(echo_handler)
        self.__dispatcher.add_handler(start_handler)
        self.__dispatcher.add_handler(MessageHandler(Filters.document.category('audio/'), self.__wav_downloader))

    def run(self):
        self.__updater.start_polling()
        self.__updater.idle()

    def __start(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=self.__greeting)

    def __send_report(self, update, context):
        self.__out_path.unlink()

        context.bot.send_message(chat_id=update.effective_chat.id, text=self.__report)

    def __process_file(self) -> str:
        _audio_decoder(self.__in_path, write_path=self.__out_path)

        self.__in_path.unlink()

        return self.__processor.process(self.__out_path)

    def __wav_downloader(self, update, context):
        context.bot.get_file(update.message.document).download()
        context.bot.send_message(chat_id=update.effective_chat.id, text='File for analysis received!')

        with open(self.__in_path, 'wb') as f:
            context.bot.get_file(update.message.document).download(out=f)

        context.bot.send_message(chat_id=update.effective_chat.id, text='File saved!')
        context.bot.send_message(chat_id=update.effective_chat.id, text='Starting processing...')

        self.__report = self.__process_file()
        self.__send_report(update, context)

    @staticmethod
    def __echo(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)
