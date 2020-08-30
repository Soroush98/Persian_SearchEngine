import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as np
import codecs
import re
from hazm import *
import math
import collections
import matplotlib.pyplot as plt
import random
from Normalize import Normalize
Norm = Normalize
class plot:
    def __init__(self):
        self.tokens = []
        self.M = []
        self.Heap_line = []
        self.cf_list = []
        self.cf_accuall = []
        self.nums = []
        self.file_names = ['ir-news-0-2.csv','ir-news-2-4.csv','ir-news-4-6.csv','ir-news-6-8.csv','ir-news-8-10.csv','ir-news-10-12.csv']
    def initialize_firstinverted(self, samples):
        dictionary = dict()
        frequency = dict()
        cf_formula = dict()

        Token_sum = 1
        b = 0.5
        k = 25
        file_choise = random.randrange(6)
        file = np.read_csv(self.file_names[file_choise])
        for i in range(samples):
            doc_choise = random.randrange(len(file))
            compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
            clean_text = BeautifulSoup(file['content'][doc_choise], "html.parser").getText()
            regex = re.compile('[a-zA-Z]')
            clean_text = regex.sub(' ', clean_text)
            junk_chars_regex = r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF\u200c]'
            remove_junk_characters = (junk_chars_regex, ' ')
            compiled_patterns_after = compile_patterns([remove_junk_characters])
            for pattern, repl in compiled_patterns_after:
                clean_text = pattern.sub(repl, clean_text)

            clean_text = Norm.normalization(clean_text)
            clean_text = clean_text.replace('\u200c', " ")

            words = clean_text.split()
            Token_sum = Token_sum + len(words)
            self.tokens.append(math.log10(Token_sum))
            self.Heap_line.append(b * math.log10(Token_sum) + math.log10(k))

            for w in words:
                if (w not in frequency):
                    frequency[w] = 0
                    frequency[w] = frequency[w] + 1
                else:
                    frequency[w] = frequency[w] + 1
                if (w not in dictionary):
                    dictionary[w] = set()
                    dictionary[w].add(doc_choise)
                else:
                    dictionary[w].add(doc_choise)
            self.M.append(math.log10(len(dictionary)))
        dictionary = Norm.remove_stopwords(dictionary)
        frequency = Norm.remove_stopwords(frequency)
        # for i in (dictionary):
        #   print(i +" " + str(sorted(dictionary[i])) )
        frequency = dict(sorted(frequency.items(), key=lambda kv: kv[1]))
        K = 0
        Sum = 0
        for i, key in enumerate(frequency.keys()):
            Sum = Sum + frequency[key]
            if (i == len(frequency) - 1):
                Sum = Sum + frequency[key]
                K = frequency[key]
                break


        for i, key in enumerate(frequency.keys()):
            self.cf_accuall.append(math.log10(frequency[key]))
            cf_formula[key] = math.log10(K / (len(frequency) + 1 - (i + 1)))
            self.cf_list.append(math.log10(K / (len(frequency) + 1 - (i + 1))))
            self.nums.append(math.log10(len(frequency) + 1 - (i + 1)))
    def initialize_secondinverted(self,samples):
        dictionary = dict()
        frequency = dict()
        cf_formula = dict()
        file = np.read_csv('ir-news-2-4.csv')

        stem_end = ['ات', 'ان', 'ترین', 'تر', 'ش', 'یی', 'ها', 'ٔ', '‌ا', '']
        start = ['می ']
        end = [' یمان', ' یم', ' یش', ' یشان', ' یتان', ' ام']

        lemm = Lemmatizer()
        stem = Stemmer()
        Token_sum = 1
        file_choise = random.randrange(6)
        file = np.read_csv(self.file_names[file_choise])
        b = 0.5
        k = 23
        for i in range(samples):
            if (i%5000 == 0):
                file_choise = random.randrange(6)
                file = np.read_csv(self.file_names[file_choise])
            doc_choise = random.randrange(len(file))
            compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
            clean_text = BeautifulSoup(file['content'][doc_choise], "html.parser").getText()
            regex = re.compile('[a-zA-Z]')
            clean_text = regex.sub(' ', clean_text)
            junk_chars_regex = r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF\u200c]'
            remove_junk_characters = (junk_chars_regex, ' ')
            compiled_patterns_after = compile_patterns([remove_junk_characters])
            for pattern, repl in compiled_patterns_after:
                clean_text = pattern.sub(repl, clean_text)

            clean_text = Norm.normalization(clean_text)
            clean_text = clean_text.replace('\u200c', " ")
            clean_text = Normalize.fix_phrases(clean_text)

            for x in start:
                y = x.replace(" ", '\u200c')
                clean_text = clean_text.replace(x, y)
            for x in end:
                y = x.replace(" ", '\u200c')
                clean_text = clean_text.replace(x, y)

            words = clean_text.split()
            Token_sum = Token_sum + len(words)
            self.tokens.append(math.log10(Token_sum))
            self.Heap_line.append(b * math.log10(Token_sum) + math.log10(k))

            for w in words:
                f = 0
                for end in stem_end:
                    if w.endswith(end):
                        w = stem.stem(w)
                        f = 1
                        break

                w = lemm.lemmatize((w)).split("#")[0]
                if (w not in frequency):
                    frequency[w] = 0
                    frequency[w] = frequency[w] + 1
                else:
                    frequency[w] = frequency[w] + 1
                if (w not in dictionary):
                    dictionary[w] = set()
                    dictionary[w].add(doc_choise)
                else:
                    dictionary[w].add(doc_choise)
            self.M.append(math.log10(len(dictionary)))
        dictionary = Norm.remove_stopwords(dictionary)
        frequency = Norm.remove_stopwords(frequency)
        frequency = dict(sorted(frequency.items(), key=lambda kv: kv[1]))
        K = 0
        Sum = 0
        for i, key in enumerate(frequency.keys()):
            Sum = Sum + frequency[key]
            if (i == len(frequency) - 1):
                Sum = Sum + frequency[key]
                K = frequency[key]
                break


        for i, key in enumerate(frequency.keys()):
            self.cf_accuall.append(math.log10(frequency[key]))
            cf_formula[key] = math.log10(K / (len(frequency) + 1 - (i + 1)))
            self.cf_list.append(math.log10(K / (len(frequency) + 1 - (i + 1))))
            self.nums.append(math.log10(len(frequency) + 1 - (i + 1)))

    def plotheap(self,):

        plt.plot(self.tokens, self.M)
        plt.plot(self.tokens, self.Heap_line)
        plt.ylabel('LogM')
        plt.show()
    def plotzipf(self):
      plt.plot(self.nums,self.cf_list)
      plt.plot(self.nums,self.cf_accuall)
      plt.ylabel('LogCf')
      plt.show()