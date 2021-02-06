import os
import tempfile
import wave

import torch
from pyannote.audio.utils.signal import Binarize
from scipy.io import wavfile


class VADException(Exception):
    pass


class VoiceActivityDetection:

    def __init__(self, binarize_params=None):
        self.sad = torch.hub.load('pyannote/pyannote-audio', model='sad_ami')
        # см. VAD Smoothing в статье https://www.isca-speech.org/archive/
        # interspeech_2015/papers/i15_2650.pdf
        binarize_params_default = {
            # an onset and offset thresholds for the detection of
            # the beginning and end of a speech segment
            'offset': 0.5,
            'onset': 0.5,
            # a threshold for small silence deletion
            'min_duration_off': 0.1,
            # a threshold for short speech segment deletion;
            'min_duration_on': 0.1,
            'log_scale': True,
        }
        binarize_params = binarize_params or binarize_params_default
        self.binarize = Binarize(**binarize_params)

    @staticmethod
    def _validate_wav_file(file_path):
        try:
            with wave.open(file_path, 'rb') as f:
                if f.getnchannels() != 2:
                    raise VADException(
                        'Invalid number of channels for wav file. Must be 2.'
                    )
        except wave.Error as e:
            raise VADException(f'Invalid format of wav file: {e}')

    @staticmethod
    def _prepare_wav_by_channels(
        source_wav, operator_channel, client_channel, tmpdir
    ):
        rate, data = wavfile.read(source_wav)
        operator_data = data[:, operator_channel]
        client_data = data[:, client_channel]

        operator_file_path = os.path.join(tmpdir, 'operator.wav')
        client_file_path = os.path.join(tmpdir, 'client.wav')

        wavfile.write(operator_file_path, rate, operator_data)
        wavfile.write(client_file_path, rate, client_data)

        return operator_file_path, client_file_path

    def _get_timeline(self, file_path):
        sad_scores = self.sad({'uri': 'filename', 'audio': file_path})
        speech = self.binarize.apply(sad_scores, dimension=0)
        return speech.for_json()['content']

    def get_timelines(self, file_path, operator_channel):
        """
        Для двухканального wav-файла возвращает разметку/таймлайн разговора
        оператора с клиентом.

        :note:
            Предполагается, что оператор и клиент разнесены по двум разным
            каналам wav-файла.

        :param file_path:
            `str`, путь до исходного wav-файла.
        :param operator_channel:
            `int`, номер канала, который относится к оператору.

        :return:
            `dict`, словарь разметки вида:
            {
                'operator_timeline': [
                    {'start': 10.5, 'end': '12.1'},
                    ...
                ],
                'client_timeline': [
                    {'start': 13, 'end': '20'},
                    ...
                ]
            }
            где параметры `start` и `end` указаны в секундах.
        """
        if operator_channel not in (0, 1):
            raise VADException('Invalid number of operator channel')

        client_channel = 0 if operator_channel else 1

        self._validate_wav_file(file_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            operator_wav, client_wav = self._prepare_wav_by_channels(
                file_path, operator_channel, client_channel, tmpdir
            )
            return {
                'operator_timeline': self._get_timeline(operator_wav),
                'client_timeline': self._get_timeline(client_wav),
            }
