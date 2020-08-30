import re
import codecs
import re
from hazm import *
import math
import collections
import matplotlib.pyplot as plt
class Normalize:
    @staticmethod
    def remove_stopwords(dictionary):
        with codecs.open("stop_words.txt", encoding='utf8') as stopwords_file:
            list = []
            for line in stopwords_file:
                list.append(Normalize.normalization(str(line.split()[0])))
            for line in list:
                if line in dictionary.keys():
                    dictionary.pop(line)
            if '' in dictionary:
                dictionary.pop('')
        return dictionary
    @staticmethod
    def normalization(text):
        text = text.lower()
        text = Normalize.remove_emoji(text)
        text = Normalize.char_translation(text)
        text = Normalize.fix_space(text)
        return text

    @staticmethod
    def char_translation(text):
        source, destination = '“”آ ىكي', '""ا یکی'
        source += "\u200b\u200d_-,.\n"
        destination += "\u200c\u200c     "
        must_remove = "۰۱۲۳۴۵۶۷۸۹٪ًٌٍَُِّـ0123456789%;:,؛،'#\\/"
        source += must_remove
        destination += ''.join(' ' for i in range(len(must_remove)))
        maketrans = lambda A, B: dict((ord(a), b) for a, b in zip(A, B))
        translator = maketrans(source, destination)
        return text.translate(translator)

    @staticmethod
    def remove_emoji(text):
        remove_range = (128512, 128591)
        for char_code in range(remove_range[0], remove_range[1] + 1):
            c = chr(char_code)
            text = text.replace(c, "")
        return text

    @staticmethod
    def fix_phrases(text):
        phrases = set()
        with codecs.open("phrase.txt", encoding='utf8') as file:
            for cnt, line in enumerate(file):
                line = line.replace("\r\n", "")
                phrases.add(line)
        for phrase in phrases:
            if (phrase in text):
                text = text.replace(phrase, Normalize.half_space_replacement(phrase))
            text = Normalize.fix_space(text)
        return text

    @staticmethod
    def fix_space(text):
        text = re.sub("\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def half_space_replacement(ph):
        ph = ph.replace(" ", '\u200c')
        return ph
