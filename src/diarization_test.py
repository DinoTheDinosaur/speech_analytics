
from diarization import runDiarization_wrapper

#Location of mono file
audio_file_path = './diarization/pyBK/audio/in_mono.wav'

#if we saving saving stereo - printing path to it
output_file = runDiarization_wrapper(audio_file_path)
#print("Location of stereo file: " + output_file)