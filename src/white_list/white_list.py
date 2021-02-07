import json
import re

from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer


def search_phrase(input_text: str, phrase: str) -> bool:
    input_text = re.sub(r'[^\w\s]', '', input_text)
    phrase = re.sub(r'[^\w\s]', '', phrase)

    morph = MorphAnalyzer()
    input_words = [morph.parse(word)[0].normal_form for word in input_text.split()]
    phrase_words = [morph.parse(word)[0].normal_form for word in phrase.split()]
    input_text = ' '.join(input_words)
    phrase = ' '.join(phrase_words)

    return input_text.find(phrase) != -1


class WhiteCheck:
    def __init__(self, file_with_list: str):
        """
        :param file_with_list: filename; file with the json extension where the white list is saved
        """
        with open(file_with_list, "r", encoding='utf-8') as file:
            self.list = json.load(file)

    @staticmethod
    def search_one_syn_phrase(input_text: str, syn_phrases: list) -> bool:
        """
        :param input_text: input text of operator's speech
        :param syn_phrases: list of dictionaries with keys: 'phrase' which means one of synonymous phrases, type of
        value is str; and 'keywords' which means str with all keywords in phrase, type of value is str;
        """
        output = False
        for dct in syn_phrases:
            if dct['keywords'] == 'all':
                output |= search_phrase(input_text, dct['phrase'])
                continue

            words = word_tokenize(dct['keywords'], language="russian")
            all_keywords_in_text = True
            for word in words:
                all_keywords_in_text &= search_phrase(input_text, word)
            output |= all_keywords_in_text

        return output

    def update_list(self, arr_of_syn_sent: list, file_with_list: str):
        """
        :param arr_of_syn_sent: list of dictionaries with keys: 'phrase' which means one of synonymous phrases, type
        of value is str; and 'keywords' which means str with all keywords in phrase, type of value is str;
        :param file_with_list: filename; file with the json extension where the white list is saved
        """
        self.list.append(arr_of_syn_sent)
        with open(file_with_list, "w", encoding='utf-8') as file:
            json.dump(self.list, file, ensure_ascii=False)

    def count_white_phrases(self, input_text: str) -> list:
        out_list = []
        for arr in self.list:
            out_list.append(int(self.search_one_syn_phrase(input_text, arr)))

        return out_list
