import json
from pathlib import Path
from typing import Optional

import nltk
from scipy.io import wavfile
from vosk import Model

from src.asr import recognize
from src.black_list import Blacklist
from src.diarization import runDiarization_wrapper
from src.id_channel import identify_operator
from src.noise_suppression import NeuralNetworkNoiseSuppressor
from src.pause_and_interruption import interruption_detection, pause_detection
from src.vad import VoiceActivityDetection
from src.white_list import WhiteCheck
from src.yandex_speech_kit import yandex_speech


class AudioProcessor:
    def __init__(self, suppressor_model_weights: Path, vosk_model: Path, white_list: Path, obscene_corpus: Path,
                 threats_corpus: Path, white_checklist: Path, black_checklist: Path,
                 recognition_engine: str = 'vosk', bucket: Optional[str] = None, aws_key: Optional[str] = None,
                 aws_key_id: Optional[str] = None, ya_api_key: Optional[str] = None) -> None:
        self.__suppressor = NeuralNetworkNoiseSuppressor(suppressor_model_weights)
        self.__vosk_model = Model(str(vosk_model))
        self.__white_checker = WhiteCheck(white_list)
        self.__black_checker = Blacklist(obscene_corpus, threats_corpus)
        self.__white_checklist = white_checklist
        self.__black_checklist = black_checklist
        self.__rec_engine = recognition_engine
        self.__bucket = bucket
        self.__aws_key = aws_key
        self.__aws_key_id = aws_key_id
        self.__ya_api_key = ya_api_key

        nltk.download('punkt')
        nltk.download('stopwords')

    def process(self, audio_path: Path) -> str:
        diarization_path = Path(__file__).parent.parent / 'diarization'

        # suppress noise
        clean, sr = self.__suppressor.suppress(audio_path, None)

        # diarize
        diarized_path, left, right, signal = runDiarization_wrapper(audio_path, clean, sr, diarization_path)
        left_wav_path = Path('left.wav')
        right_wav_path = Path('right.wav')

        wavfile.write(left_wav_path, sr, left)
        wavfile.write(right_wav_path, sr, right)

        # speech recognition
        if self.__rec_engine == 'vosk':
            left_text = recognize(self.__vosk_model, str(left_wav_path))
            right_text = recognize(self.__vosk_model, str(right_wav_path))
        else:
            left_text = yandex_speech(left_wav_path, self.__bucket, self.__aws_key, self.__aws_key_id,
                                      self.__ya_api_key)
            right_text = yandex_speech(left_wav_path, self.__bucket, self.__aws_key, self.__aws_key_id,
                                       self.__ya_api_key)

        left_wav_path.unlink(missing_ok=True)
        right_wav_path.unlink(missing_ok=True)

        op_channel_num = int(identify_operator(left_text, right_text))
        op_text = left_text if op_channel_num == 0 else right
        output = {}

        with open(self.__white_checklist, 'r', encoding='utf-8') as f:
            white_weights = json.load(f)
            count_list = self.__white_checker.count_white_phrases(op_text)

            for i in range(len(white_weights)):
                output[white_weights[i][0]] = count_list[i] * white_weights[i][1]

        with open(self.__black_checklist, 'r', encoding='utf-8') as f:
            black_weights = json.load(f)
            count_dict = self.__black_checker.bad_words(op_text)

            for key, value in count_dict.items():
                output[black_weights[key][0]] = value * black_weights[key][1]

        vad = VoiceActivityDetection()
        markup = vad.get_timelines(str(diarized_path), op_channel_num)

        output['Число перебиваний'] = -interruption_detection(audio_path, markup)
        output['Суммарная оценка'] = sum(output.values())
        output['Число перебиваний'] = -output['Число перебиваний']
        output['Средняя длина паузы оператора'] = pause_detection(markup)

        return '\n'.join([f'{k}:   {v}' for k, v in output.items()])
