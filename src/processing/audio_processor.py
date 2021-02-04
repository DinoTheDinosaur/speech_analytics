from pathlib import Path
from pprint import pprint

from src.diarization import runDiarization_wrapper
from src.noise_suppression import NeuralNetworkNoiseSuppressor
from src.vad.vad_pyannote import VoiceActivityDetection

SAMPLE_RATE: int = 16000


class AudioProcessor:
    def __init__(self, model_weights_path: Path) -> None:
        self.__suppressor = NeuralNetworkNoiseSuppressor(model_weights_path)

    def process(self, audio_path: Path) -> int:
        diarization_path = Path(__file__).parent.parent / 'diarization'
        clean, sr = self.__suppressor.suppress(audio_path, None)

        diarized = runDiarization_wrapper(audio_path, clean, sr, diarization_path)

        # # TODO: operator channel is unknown for now
        vad = VoiceActivityDetection()
        markup = vad.get_timelines(str(diarized), 0)

        pprint(markup)

        return 0


ap = AudioProcessor(Path(r''))
ap.process(Path(r''))
