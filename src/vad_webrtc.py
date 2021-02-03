import os
import wave
import tempfile

from dataclasses import dataclass
from collections import deque

import webrtcvad

from scipy.io import wavfile


class VADException(Exception):
    pass


@dataclass
class Frame:
    frame_bytes: bytes
    timestamp: float
    duration: float


class VoiceActivityDetection:

    def __init__(
        self, filtering_mode=0, frame_duration_ms=30, padding_duration_ms=300,
        threshold_percent=0.9
    ):
        if filtering_mode not in (0, 1, 2):
            raise VADException('Invalid filtering mode')

        self.vad = webrtcvad.Vad(mode=filtering_mode)
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.threshold_percent = threshold_percent

    @staticmethod
    def _validate_wav_file(file_path):
        try:
            with wave.open(file_path, 'rb') as f:
                if f.getnchannels() != 2:
                    raise VADException(
                        'Invalid number of channels for wav file. Must be 2.'
                    )
                if f.getsampwidth() != 2:
                    raise VADException(
                        'Invalid sample width in bytes. Must be 2.'
                    )
                if f.getframerate() not in (8000, 16000, 32000, 48000):
                    raise VADException('Invalid frame rate.')
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

    def _get_frames(self, pcm_data, sample_rate):
        n = int(sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        frames = []
        while offset + n < len(pcm_data):
            frames.append(
                Frame(pcm_data[offset:offset + n], timestamp, duration)
            )
            timestamp += duration
            offset += n
        return frames

    def _vad_collector(self, frames, sample_rate):
        num_padding_frames = int(
            self.padding_duration_ms / self.frame_duration_ms
        )
        ring_buffer = deque(maxlen=num_padding_frames)

        timeline = []
        triggered = False
        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)
            ring_buffer.append((frame, is_speech))

            if not triggered:
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > self.threshold_percent * ring_buffer.maxlen:
                    timeline.append({
                        'start': round(ring_buffer[0][0].timestamp, 2)
                    })
                    ring_buffer.clear()
                    triggered = True
            else:
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech]
                )
                if num_unvoiced > self.threshold_percent * ring_buffer.maxlen:
                    timeline[-1]['stop'] = round(
                        frame.timestamp + frame.duration, 2
                    )
                    ring_buffer.clear()
                    triggered = False

        if triggered:
            timeline[-1]['stop'] = round(frame.timestamp + frame.duration, 2)

        return timeline

    def _get_timeline(self, file_path):
        with wave.open(file_path, 'rb') as f:
            sample_rate = f.getframerate()
            pcm_data = f.readframes(f.getnframes())

        frames = self._get_frames(pcm_data, sample_rate)
        return self._vad_collector(frames, sample_rate)

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
