from vosk import Model, KaldiRecognizer, SetLogLevel
from src.vosk_asr.config import MODEL_PATH
import wave
import json


SetLogLevel(-1)
model = Model(MODEL_PATH)


def recognize(wav_file_path):
    """
    Speech to text recognizer for russian speech using vosk models
    path to russian vosk model should be configured in config.py file
    """
    with wave.open(wav_file_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise TypeError("Audio file must be WAV format mono PCM.")
        rec = KaldiRecognizer(model, wf.getframerate())
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        json_ = json.loads(rec.FinalResult())
        return json_['text']
