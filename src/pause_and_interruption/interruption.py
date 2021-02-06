import numpy as np
import torch
from pyannote.audio.utils.signal import Binarize

ovl = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami')


def interruption_detection(mono_file, vad_dictionary, ovl=ovl, sucessfull_identification=1, delay=0.2):
    """
    на вход принимает:
    mono_file - запись wav до диаризации,
    vad_dictionary - результат работы VAD,
    sucessful_identification - удалось ли определить канал оператора,
    delay - допустимая задержка телефонной связи

    пример вызова:
    interruption_detection('clean_1.wav', vad_activity)

    возвращает целое число - кол-во перебиваний
    """
    test_file = {'uri': '1', 'audio': mono_file}
    ovl_scores = ovl(test_file)
    binarize = Binarize(offset=0.55, onset=0.55, log_scale=True,
                        min_duration_off=0.1, min_duration_on=0.1)
    overlap = binarize.apply(ovl_scores, dimension=1)
    overlap = dict(overlap.for_json())

    interruption_count = 0

    if sucessfull_identification == 1:
        for one_overlap in overlap['content']:
            start_interrupt = one_overlap['start']
            for client_speech in vad_dictionary['client_timeline']:
                if (start_interrupt > client_speech['start']) and (start_interrupt < client_speech['end']):
                    interruption_count += 1

        client_activity, operator_activity = change_dict_format(vad_dictionary)
        for end in client_activity[1]:
            difference = operator_activity[0] - end
            interruption_count += difference[np.abs(difference) < delay].shape[0]

    return interruption_count


def change_dict_format(vad_dictionary):
    """"
    меняет формат словаря с VAD клиента и оператора
    {'end': 20.93515625, 'start': 20.34284375},
    {'end': 21.510593749999998, 'start': 21.127531249999997},...
    в двумерный numpy массив array([[20.34284375, 21.1275312499997], [20.93515625, 21.510593749999998]])
    """
    client_activity = [[], []]
    operator_activity = [[], []]

    for timeline in vad_dictionary['client_timeline']:
        client_activity[0].append(timeline['start'])
        client_activity[1].append(timeline['end'])

    for timeline in vad_dictionary['operator_timeline']:
        operator_activity[0].append(timeline['start'])
        operator_activity[1].append(timeline['end'])

    client_activity = np.array(client_activity)
    operator_activity = np.array(operator_activity)

    return client_activity, operator_activity
