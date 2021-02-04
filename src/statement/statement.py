import json

from src.black_list.black_list import Blacklist
from src.id_channel.identification import identify_operator
from src.statement import conf
from src.vosk_asr.vosk_asr import recognize
from src.white_list.white_list import WhiteCheck


def evaluate_operator(file_path1: str, file_path2: str) -> dict:
    output = {}

    text1 = recognize(file_path1)
    text2 = recognize(file_path2)

    operator_text = text2 if identify_operator(text1, text2) else text1

    white_checker = WhiteCheck(conf.WHITE_LIST)
    with open(conf.CHECK_LIST_WHITE, "r", encoding='utf-8') as file:
        white_weights = json.load(file)
        count_list = white_checker.count_white_phrases(operator_text)
        for i in range(len(white_weights)):
            output[white_weights[i][0]] = count_list[i] * white_weights[i][1]

    black_checker = Blacklist(conf.OBSCENE_CORPUS, conf.THREATS_CORPUS)
    with open(conf.CHECK_LIST_BLACK, "r", encoding='utf-8') as file:
        black_weights = json.load(file)
        count_dict = black_checker.bad_words(operator_text)
        for key, value in count_dict.items():
            output[black_weights[key][0]] = value * black_weights[key][1]

    # TODO: "Долг" и "просрочка" (?), перебивания

    return output
