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
        if self.__rec_engine == 'yandex':
            try:
                left_text = yandex_speech(str(left_wav_path), self.__bucket, self.__aws_key,
                                          self.__aws_key_id,
                                          self.__ya_api_key)
            except TimeoutError:
                print('YSK recognition failed for left channel, using vosk')
                left_text = recognize(self.__vosk_model, str(left_wav_path))

            try:
                right_text = yandex_speech(str(right_wav_path), self.__bucket, self.__aws_key,
                                           self.__aws_key_id,
                                           self.__ya_api_key)
            except TimeoutError:
                print('YSK recognition failed for right channel, using vosk')
                right_text = recognize(self.__vosk_model, str(right_wav_path))
        else:
            left_text = recognize(self.__vosk_model, str(left_wav_path))
            right_text = recognize(self.__vosk_model, str(right_wav_path))

        left_wav_path.unlink(missing_ok=True)
        right_wav_path.unlink(missing_ok=True)

        op_channel_num = int(identify_operator(left_text, right_text))
        op_text = left_text if op_channel_num == 0 else right_text
        output = {}

        with open(self.__white_checklist, 'r', encoding='utf-8') as f:
            white_weights = json.load(f)
            count_list = self.__white_checker.count_white_phrases(op_text)

            for i in range(len(white_weights)):
                if count_list[i] >= 1:
                    estimate = 0
                else:
                    estimate = white_weights[i][1]

                output[white_weights[i][0]] = {
                    'description': white_weights[i][0],
                    'weight': white_weights[i][1],
                    'count': count_list[i],
                    'estimate': estimate
                }

        with open(self.__black_checklist, 'r', encoding='utf-8') as f:
            black_weights = json.load(f)
            count_dict = self.__black_checker.bad_words(op_text)

            for key, value in count_dict.items():
                output[black_weights[key][0]] = {
                    'description': black_weights[key][0],
                    'weight': black_weights[key][1],
                    'count': value,
                    'estimate': black_weights[key][1] * value
                }

        vad = VoiceActivityDetection()
        markup = vad.get_timelines(str(diarized_path), op_channel_num)

        num_interruption = interruption_detection(audio_path, markup)
        output['Перебивания'] = {
            'description': "Перебивания",
            'weight': 1,
            'count': num_interruption,
            'estimate': num_interruption
        }

        glob_estimate = 0

        for key in output.keys():
            glob_estimate += output[key]['estimate']

        res = '\n'.join([f'{k}:   {output[k]["count"]}' for k in output])

        output['Средняя длина паузы оператора'] = pause_detection(markup)
        output['Итоговая оценка'] = glob_estimate

        res += '\nСредняя длина паузы:   ' + str(output['Средняя длина паузы оператора']) + \
               '\nИтоговая оценка:   ' + str(output['Итоговая оценка'])

        return res
