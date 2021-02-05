import json
from pathlib import Path
from pprint import pprint
from typing import Dict

import nltk
from scipy.io import wavfile
from vosk import Model

from src.asr.vosk_asr import recognize
from src.black_list.black_list import Blacklist
from src.diarization import runDiarization_wrapper
from src.id_channel.identification import identify_operator
from src.noise_suppression import NeuralNetworkNoiseSuppressor
from src.vad.vad_pyannote import VoiceActivityDetection
from src.white_list.white_list import WhiteCheck

SAMPLE_RATE: int = 16000


class AudioProcessor:
    def __init__(self, suppressor_model_weights: Path, vosk_model: Path, white_list: Path, obscene_corpus: Path,
                 threats_corpus: Path, white_checklist: Path, black_checklist: Path) -> None:
        self.__suppressor = NeuralNetworkNoiseSuppressor(suppressor_model_weights)
        self.__vosk_model = Model(str(vosk_model))
        self.__white_checker = WhiteCheck(white_list)
        self.__black_checker = Blacklist(obscene_corpus, threats_corpus)
        self.__white_checklist = white_checklist
        self.__black_checklist = black_checklist

        nltk.download('punkt')
        nltk.download('stopwords')

    def process(self, audio_path: Path) -> Dict:
        diarization_path = Path(__file__).parent.parent / 'diarization'
        clean, sr = self.__suppressor.suppress(audio_path, None)

        diarized_path, left, right, signal = runDiarization_wrapper(audio_path, clean, sr, diarization_path)
        left_wav_path = Path('left.wav')
        right_wav_path = Path('right.wav')

        wavfile.write(left_wav_path, sr, left)
        wavfile.write(right_wav_path, sr, right)

        left_text = recognize(self.__vosk_model, str(left_wav_path))
        left_wav_path.unlink(missing_ok=True)

        right_text = recognize(self.__vosk_model, str(right_wav_path))
        right_wav_path.unlink(missing_ok=True)

        print(f'LEFT CHANNEL TEXT: {left_text}')
        print(f'RIGHT CHANNEL TEXT: {right_text}')

        op_channel_num = int(identify_operator(left_text, right_text))
        op_audio = left if op_channel_num == 0 else right
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

        print('VAD')
        pprint(markup)

        return output
