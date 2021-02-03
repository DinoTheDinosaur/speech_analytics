
from diarization import runDiarization_wrapper

#Location of mono file
audio_file_path = './diarization/pyBK/audio/vaneatian2.wav'
output_file = runDiarization_wrapper(audio_file_path)
print("Location of stereo file: " + output_file)