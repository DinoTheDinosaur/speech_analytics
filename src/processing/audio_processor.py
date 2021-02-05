from pathlib import Path
from pprint import pprint

from vosk import Model
from scipy.io import wavfile

from src.asr.vosk_asr import recognize
from src.diarization import runDiarization_wrapper
from src.noise_suppression import NeuralNetworkNoiseSuppressor
from src.vad.vad_pyannote import VoiceActivityDetection

SAMPLE_RATE: int = 16000


class AudioProcessor:
    def __init__(self, suppressor_model_weights_path: Path, vosk_model_path: Path) -> None:
        self.__suppressor = NeuralNetworkNoiseSuppressor(suppressor_model_weights_path)
        self.__vosk_model = Model(vosk_model_path)

    def process(self, audio_path: Path) -> int:
        diarization_path = Path(__file__).parent.parent / 'diarization'
        clean, sr = self.__suppressor.suppress(audio_path, None)

        diarized_path, left, right, signal = runDiarization_wrapper(audio_path, clean, sr, diarization_path)
        wavfile.write('left.wav', sr, left)
        wavfile.write('right.wav', sr, right)

        left_text = recognize(self.__vosk_model, 'left.wav')
        right_text = recognize(self.__vosk_model, 'right.wav')

        print(f'LEFT CHANNEL TEXT: {left_text}')
        print(f'RIGHT CHANNEL TEXT: {right_text}')

        # TODO: detect operator's channel here

        # TODO: operator channel is unknown for now
        vad = VoiceActivityDetection()
        markup = vad.get_timelines(str(diarized_path), 0)

        pprint(markup)

        return 0
