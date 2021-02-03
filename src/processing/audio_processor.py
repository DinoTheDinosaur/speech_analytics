from pathlib import Path

import soundfile as sf

from src.noise_suppression import NeuralNetworkNoiseSuppressor
from src.diarization import runDiarization_wrapper
from src.vad.vad_webrtc import VoiceActivityDetection

SAMPLE_RATE: int = 16000


class AudioProcessor:
    def __init__(self, model_weights_path: Path) -> None:
        self.__suppressor = NeuralNetworkNoiseSuppressor(model_weights_path)

    def process(self, audio_path: Path) -> int:
        # suppress noise
        clean = self.__suppressor.suppress(audio_path, SAMPLE_RATE)
        clean_audio_path = Path(__file__).parent.parent / 'diarization' / 'pyBK' / 'audio' / ('clean_' + audio_path.name)
        sf.write(clean_audio_path, clean, SAMPLE_RATE, format='WAV')

        # diarization
        runDiarization_wrapper(clean_audio_path, clean_audio_path)

        # vad
        # TODO: operator channel?
        vad = VoiceActivityDetection()
        markup = vad.get_timelines(clean_audio_path, 0)

        print(markup)

        return 0


ap = AudioProcessor(Path(r'D:\MLProjects\speech_analytics\model_weights.ckpt'))
ap.process(Path(r'D:\MLProjects\speech_analytics_data\loan_for_test\1.wav'))