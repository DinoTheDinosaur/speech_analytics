import configparser
import os

import numpy as np
from pyAudioAnalysis import audioBasicIO
from pydiarization.diarization_wrapper import rttm_to_string

from .pyBK.main import runDiarization


# helper function
def getCurrentSpeakerCut(rttmString):
    return float(rttmString.split(' ')[3]), float(rttmString.split(' ')[3]) + float(rttmString.split(' ')[4]), \
           rttmString.split(' ')[7]


def runDiarization_wrapper(showName):
    """Running Diarization from pyBK and returning stereo array in output"""

    # reading pyBK config file
    configFile = './diarization/pyBK/config.ini'
    config = configparser.ConfigParser()
    config.read(configFile)
    # extracting file name
    baseFileName = os.path.basename(showName)
    fileName = os.path.splitext(baseFileName)[0]

    # If the output file already exists from a previous call it is deleted
    if os.path.isfile(config['PATH']['output'] + config['EXPERIMENT']['name'] + config['EXTENSION']['output']):
        os.remove(config['PATH']['output'] + config['EXPERIMENT']['name'] + config['EXTENSION']['output'])

    if os.path.isfile(config['PATH']['file_output'] + config['EXPERIMENT']['name'] + config['EXTENSION']['audio']):
        os.remove(config['PATH']['file_output'] + config['EXPERIMENT']['name'] + config['EXTENSION']['audio'])

    # Output folder is created
    if not os.path.isdir(config['PATH']['output']):
        os.mkdir(config['PATH']['output'])

    # Output folder for wav file is created
    if not os.path.isdir(config['PATH']['file_output']):
        os.mkdir(config['PATH']['file_output'])

    # Start of diarization
    print('\nProcessing file', fileName)
    runDiarization(fileName, config)

    # Parsing rttm file for extraction speakers time
    rttm_file = config['EXPERIMENT']['name'] + ".rttm"
    path = "./diarization/pyBK/out/" + rttm_file
    rttmString = rttm_to_string(path)
    resArray = rttmString.split('SPEAKER')

    # Reading mono file
    sampling_rate, signal = audioBasicIO.read_audio_file(showName)
    signal = audioBasicIO.stereo_to_mono(signal)

    # arrays for left and right  channels of stereo file
    left = np.zeros(len(signal))
    right = np.zeros(len(signal))

    # extracting speech of every speaker#1  and speaker#2
    for i in range(1, len(resArray)):
        # convertion time to sample number
        begin, end, speaker = getCurrentSpeakerCut(resArray[i])
        begin = int(begin * sampling_rate)
        end = int(end * sampling_rate)

        if speaker == 'speaker1':
            left[begin:end] = signal[begin:end]
        elif speaker == "speaker2":
            right[begin:end] = signal[begin:end]
        else:
            left[begin:end] = signal[begin:end]
            right[begin:end] = signal[begin:end]

    # convert float to int - it help to avoid problems with incorrectly saved wav file
    # combine left and right signal into one 2D array
    stereo_array = np.vstack((left.astype(np.int16), right.astype(np.int16))).T

    # saving stereo file
    # output_file = config['PATH']['file_output'] + fileName + "_stereo.wav"
    # write(output_file, sampling_rate, stereo_array)

    # return stereo_array
    return stereo_array
