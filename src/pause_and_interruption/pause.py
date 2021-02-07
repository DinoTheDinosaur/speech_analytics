import numpy as np
from pyannote.core import Timeline, Segment

from src.pause_and_interruption.interruption import change_dict_format


def pause_detection(vad_dictionary):
    """
    на вход принимает vad_dictionary - результат работы модуля VAD
    возвращает среднюю продолжительность пауз оператора в сек
    """
    client_activity, operator_activity = change_dict_format(vad_dictionary)
    operator_timeline = Timeline([Segment(x['start'], x['end']) for x in vad_dictionary['operator_timeline']])
    operator_delays = []

    for client_end in client_activity[1]:
        operator_client_diff = operator_activity[0] - client_end
        operator_client_diff = operator_client_diff[operator_client_diff > 0]

        if len(operator_client_diff) > 0:
            if len(operator_timeline.overlapping(client_end)) == 0:
                operator_delays.append(operator_client_diff.min())

    return np.mean(operator_delays)
