import re
import spacy

marked_phrases = [['Вас приветствует'], ['компания'], ['Страна Экспресс'], ['разговор', 'записываться'],
                  ['разговор', 'запись'],
                  ]


def text_preprocessing(input_text: str) -> str:
    input_text = re.sub(r'[^\w\s]', '', input_text)
    input_text.lower()

    nlp = spacy.load('ru2')
    input_words = [word.lemma_ for word in nlp(input_text)]
    input_text = ' '.join(input_words)

    return input_text


def search_markers(input_text: str) -> int:
    input_text = text_preprocessing(input_text)
    output = 0

    for arr in marked_phrases:
        all_elements_in_text = True

        for marker in arr:
            marker = text_preprocessing(marker)
            all_elements_in_text &= (input_text.find(marker) != -1)

        if all_elements_in_text:
            output += 1

    return output


def identify_operator(text1: str, text2: str) -> bool:
    """
    :param text1: text of the speech of the first channel
    :param text2: text of the speech of the second channel
    :return: False if operator's channel is the first; otherwise, it returns True
    """
    return search_markers(text1) <= search_markers(text2)
