from interruption import interruption_detection
from pause import pause_detection



interruption_detection('mono.wav', vad_dictionary)
#returns 4
pause_detection(vad_dictionary)
#returns 0.3545859374999 (in sec)
