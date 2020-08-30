from bs4 import BeautifulSoup
import pandas as np
import codecs
import re
from hazm import *
import math
import collections
import matplotlib.pyplot as plt
from Normalize import Normalize
from Plot import plot
import numpy
import  operator
import pickle
import csv
class inverted:
    def __init__(self):

        self.file_names = ['ir-news-0-2.csv', 'ir-news-2-4.csv', 'ir-news-4-6.csv', 'ir-news-6-8.csv',
                           'ir-news-8-10.csv', 'ir-news-10-12.csv']
        # self.combined = "combined.csv"
        self.frequency = dict()
        self.dictionary = dict()
        self.doc_vector = dict()
        self.tf = dict()
        self.totalDocs = 0
        self.R = 10
        self.champ = dict()
    def create_first_inverted(self):
        file = np.read_csv(self.file_names[0])
        Norm = Normalize
        self.totalDocs = int(len(file['content']))
        for i in range(0, self.totalDocs):
            compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
            clean_text = BeautifulSoup(file['content'][i], "html.parser").getText()
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

            for w in words:
                if (w not in self.frequency):
                    self.frequency[w] = 0
                    self.frequency[w] = self.frequency[w] + 1
                else:
                    self.frequency[w] = self.frequency[w] + 1
                if (w not in self.dictionary):
                    self.dictionary[w] = set()
                    self.dictionary[w].add(i)
                else:
                    self.dictionary[w].add(i)

        self.dictionary = Norm.remove_stopwords(self.dictionary)
        self.frequency = Norm.remove_stopwords(self.frequency)
        return self.dictionary,self.totalDocs
    def create_second_inverted(self):
        file = np.read_csv(self.file_names[0])
        stem_end = ['ات', 'ان', 'ترین', 'تر', 'ش', 'یی', 'ها', 'ٔ', '‌ا', '']
        start = ['می ']
        end = [' یمان', ' یم', ' یش', ' یشان', ' یتان', ' ام']
        Norm = Normalize
        lemm = Lemmatizer()
        stem = Stemmer()
        tempfreq = dict()
        self.totalDocs = int(len(file['content'])/10)
        for i in range(0,self.totalDocs):
            compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
            clean_text = BeautifulSoup(file['content'][i], "html.parser").getText()
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
                clean_text = clean_text.replace(x,y)
            for x in end:
                y = x.replace(" ",'\u200c')
                clean_text = clean_text.replace(x, y)
            words = clean_text.split()

            for w in words:
                f = 0
                for end in stem_end:
                    if w.endswith(end):
                        w = stem.stem(w)
                        f = 1
                        break
                temp = w
                w = lemm.lemmatize((w)).split("#")[0]
                # if (w == "خواه"):
                #     print(temp)
                if (w not in self.frequency):
                    self.frequency[w] = 0
                    self.frequency[w] = self.frequency[w] + 1
                else:
                    self.frequency[w] = self.frequency[w] + 1
                if (w not in self.dictionary):
                    self.dictionary[w] = set()
                    self.dictionary[w].add(i)
                else:
                    self.dictionary[w].add(i)
                if (w not in self.champ):
                    self.champ[w] = dict()
                    if (i not in self.champ[w]):
                        self.champ[w][i] = 0
                        self.champ[w][i]  = self.champ[w][i] + 1
                else:
                    if (i not in self.champ[w]):
                        self.champ[w][i] = 0
                        self.champ[w][i] = self.champ[w][i] + 1

                    else:
                        self.champ[w][i] = self.champ[w][i] + 1

                if (w not in tempfreq):
                    tempfreq[w] = list()
                    tempfreq[w] = numpy.zeros(self.totalDocs)
                    tempfreq[w][i] = tempfreq[w][i] + 1
                else:
                   tempfreq[w][i] = tempfreq[w][i] + 1
        self.tf = tempfreq
        self.dictionary = Norm.remove_stopwords(self.dictionary)
        self.frequency = Norm.remove_stopwords(self.frequency)
        with open('dict.txt', 'w', encoding="utf-8") as f:
            for key in self.dictionary:
                f.write('%s,' % key)
                for j in self.dictionary[key]:
                    f.write("%s " % str(j))
                f.write("\n")

        with open('tf.txt', 'w',encoding="utf-8") as f:
            for key in self.tf:
                f.write('%s,' %key)
                for j in self.tf[key]:
                    f.write("%s "%str(j))
                f.write("\n")

        return self.dictionary,self.totalDocs,self.tf

    def create_champion(self):
        champion_index = dict()
        for w in self.champ:
            temp = self.champ[w]
            temp = sorted(temp.items(), key=operator.itemgetter(1),reverse=True)
            # if (w == 'ورزش'):
            #     print(temp)
            if (w not in champion_index):
                champion_index[w] = list()
                i = 0
                for (key,item) in temp :
                    champion_index[w].append(key)
                    i = i+1
                    if (i== self.R):
                        break
            # if (w=="ارام"):
            #     file = np.read_csv(self.file_names[0])
            #     for i in range(len(champion_index['ارام'])):
            #         print(file['title'][champion_index['ارام'][i]])
        with open('champion.txt', 'w', encoding="utf-8") as f:
            for key in champion_index:
                f.write('%s,' % key)
                for j in champion_index[key]:
                    f.write("%s " % str(j))
                f.write("\n")
        return champion_index
            # if (w == 'ورزش'):
            #     print(champion_index['ورزش'])
        # file = np.read_csv(self.file_names[0])
        # for i in range(len(champion_index['ورزش'])):
        #     print("heaet")
        #     print(file['content'][champion_index['ورزش'][i]])










inv = inverted()
# dictionary = inv.create_first_inverted()
# for w in (dictionary):
#     print(w + " " + str(dictionary[w]))
dictionary = inv.create_second_inverted()
inv.create_champion()


# for w in (dictionary):
#     print(w + " " + str(dictionary[w]))
# plotobj = plot()
# plotobj.initialize_firstinverted(15000)
# plotobj.plotheap()
# plotobj.plotzipf()
