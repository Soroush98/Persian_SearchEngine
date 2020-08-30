import re
from bs4 import BeautifulSoup
from Normalize import Normalize
from hazm import *
import numpy
import math
import pickle
from  Inverted  import inverted
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import operator
import numpy as np
import  pandas
from Max_heap import MaxHeap
class query_processor:
    def __init__(self  ):
        self.file_names = ['ir-news-0-2.csv', 'ir-news-2-4.csv', 'ir-news-4-6.csv', 'ir-news-6-8.csv',
                           'ir-news-8-10.csv', 'ir-news-10-12.csv']
        # self.combined = "combined.csv"
        file = pandas.read_csv(self.file_names[0])

        self.totalDocs = int(len(file['content'])/10)
        self.doc_vector = dict()
        self.frequency = dict()
        self.query_vector = list()
        self.tf = dict()
        self.K = 10
        # f = open('champion.txt', 'r',encoding="utf-8")
        # self.champion = f.read()
        # f.close()
        self.index_tf = dict()
        with open('tf.txt', 'r', encoding="utf-8") as f:
            k=0
            while (f.readable()):
                temp = f.readline()
                temp2 = temp.split(",")
                if (len(temp2) == 1):
                    break
                temp3 = temp2[1].split(" ")
                if (temp2[0] not in self.index_tf):
                    self.index_tf[temp2[0]] = numpy.zeros(self.totalDocs)
                    for i in range(self.totalDocs):
                        if ( temp3[i]!= "\n"):
                            self.index_tf[temp2[0]][i] = float(temp3[i])
        self.dictionary = dict()
        with open('dict.txt', 'r', encoding="utf-8") as f:
            k=0
            while (f.readable()):
                temp = f.readline()
                temp2 = temp.split(",")
                if (len(temp2) == 1):
                    break
                temp3 = temp2[1].split(" ")
                if (temp2[0] not in self.dictionary):
                    self.dictionary[temp2[0]] = set()
                    for i in range(len(temp3)):
                        if ( temp3[i]!= "\n"):
                            self.dictionary[temp2[0]].add( int(temp3[i]))
        self.champion = dict()
        with open('champion.txt', 'r', encoding="utf-8") as f:
            k=0
            while (f.readable()):
                temp = f.readline()
                temp2 = temp.split(",")
                if (len(temp2) == 1):
                    break
                temp3 = temp2[1].split(" ")
                if (temp2[0] not in self.champion):
                    self.champion[temp2[0]] = list()
                    for i in range(len(temp3)):
                        if ( temp3[i]!= "\n"):
                            self.champion[temp2[0]].append(int(temp3[i]))
        self.champion_doc_vector = dict()
    def champion_process(self):
        stem_end = ['ات', 'ان', 'ترین', 'تر', 'ش', 'یی', 'ها', 'ٔ', '‌ا', '']
        start = ['می ']
        end = [' یمان', ' یم', ' یش', ' یشان', ' یتان', ' ام']
        Norm = Normalize
        lemm = Lemmatizer()
        stem = Stemmer()
        query = input("Enter your query :")
        start_time = time.time()
        compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
        clean_text = BeautifulSoup(query, "html.parser").getText()
        regex = re.compile('[a-zA-Z]')
        clean_text = regex.sub(' ', clean_text)
        junk_chars_regex = r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF\u200c]'
        remove_junk_characters = (junk_chars_regex, ' ')
        compiled_patterns_after = compile_patterns([remove_junk_characters])
        for pattern, repl in compiled_patterns_after:
            clean_text = pattern.sub(repl, clean_text)

        clean_text = Norm.normalization(clean_text)
        clean_text = clean_text.replace('\u200c', " ")

        for x in start:
            y = x.replace(" ", '\u200c')
            clean_text = clean_text.replace(x, y)
        for x in end:
            y = x.replace(" ", '\u200c')
            clean_text = clean_text.replace(x, y)
        words = clean_text.split()
        allwords = []
        tempfreq = dict()
        for w in self.dictionary:
            tempfreq[w] = numpy.zeros(1)
        for w in words:
            for end in stem_end:
                if w.endswith(end):
                    w = stem.stem(w)
                    break
            w = lemm.lemmatize((w)).split("#")[0]
            tempfreq[w][0] = tempfreq[w][0] + 1
            allwords.append(w)
        self.tf = tempfreq
        print(words)
        self.frequency = Norm.remove_stopwords(self.frequency)
        self.query_vector = list()
        for w in self.dictionary:
            if (self.tf[w][0] != 0):
                self.query_vector.append((1 + math.log10(self.tf[w][0])) * math.log10(
                    self.totalDocs / len(self.dictionary[w])))
            else:
                self.query_vector.append(0)
        self.champion_vector_space(allwords)
        self.sort_champion()
        print("--- %s seconds ---" % (time.time() - start_time))
    def champion_vector_space(self,words):
        doc_list = set()
        for w in words:
            for j in range(len(self.champion[w])):
                doc_list.add(self.champion[w][j])
        for i in doc_list:
            ind = i
            for w in self.dictionary:
                if (ind not in self.champion_doc_vector):
                    self.champion_doc_vector[ind] = list()
                    if (self.index_tf[w][ind] != 0):
                        self.champion_doc_vector[ind].append((1+ math.log10(self.index_tf[w][ind]))* math.log10(self.totalDocs / len(self.dictionary[w])))
                    else:
                        self.champion_doc_vector[ind].append(0)
                else:
                    if (self.index_tf[w][ind] != 0):
                        self.champion_doc_vector[ind].append((1+ math.log10(self.index_tf[w][ind]))* math.log10(self.totalDocs / len(self.dictionary[w])))
                    else:
                        self.champion_doc_vector[ind].append(0)
            print(self.champion_doc_vector[ind])
    def process(self):
        stem_end = ['ات', 'ان', 'ترین', 'تر', 'ش', 'یی', 'ها', 'ٔ', '‌ا', '']
        start = ['می ']
        end = [' یمان', ' یم', ' یش', ' یشان', ' یتان', ' ام']
        Norm = Normalize
        lemm = Lemmatizer()
        stem = Stemmer()
        query = input("Enter your query :")
        start_time = time.time()
        compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
        clean_text = BeautifulSoup(query, "html.parser").getText()
        regex = re.compile('[a-zA-Z]')
        clean_text = regex.sub(' ', clean_text)
        junk_chars_regex = r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF\u200c]'
        remove_junk_characters = (junk_chars_regex, ' ')
        compiled_patterns_after = compile_patterns([remove_junk_characters])
        for pattern, repl in compiled_patterns_after:
            clean_text = pattern.sub(repl, clean_text)

        clean_text = Norm.normalization(clean_text)
        clean_text = clean_text.replace('\u200c', " ")

        for x in start:
            y = x.replace(" ", '\u200c')
            clean_text = clean_text.replace(x, y)
        for x in end:
            y = x.replace(" ", '\u200c')
            clean_text = clean_text.replace(x, y)
        words = clean_text.split()

        tempfreq = dict()
        for w in self.dictionary:
            tempfreq[w] = numpy.zeros(1)
        for w in words:
            for end in stem_end:
                if w.endswith(end):
                    w = stem.stem(w)
                    break
            w = lemm.lemmatize((w)).split("#")[0]
            tempfreq[w][0] = tempfreq[w][0] + 1
        self.tf = tempfreq
        print(words)
        self.frequency = Norm.remove_stopwords(self.frequency)
        self.query_vector = list()
        for w in self.dictionary:
            if (self.tf[w][0] != 0):
                self.query_vector.append((1 + math.log10(self.tf[w][0])) * math.log10(
                    self.totalDocs / len(self.dictionary[w])))
            else:
                self.query_vector.append(0)
        self.doc_vector = self.vector_space()
        self.sort()
        print("--- %s seconds ---" % (time.time() - start_time))
    def sort_champion(self):
        distances = dict()
        max_heap = MaxHeap(len(self.dictionary))
        count = 0
        for w in self.champion_doc_vector:
            if (norm(self.champion_doc_vector[w])!= 0):
                dist = self.calculate_sim(self.champion_doc_vector[w])
                if (dist != 0):                             #index elemination
                    distances[w] = dist                                    #max heap construct
                    max_heap.insert(distances[w])
        chosen_docs = list()
        for i in range(self.K):
            x = max_heap.extractMax()
            print(x)
            chosen_doc = [key for (key, value) in distances.items() if value == x]
            chosen_docs.append(chosen_doc[0])
            print(chosen_doc)
        # sorted_distance = sorted(distances.items(), key=operator.itemgetter(1),reverse=True)
        self.show_topK(chosen_docs)
        # print((sorted_distance))

    def sort(self):
        distances = dict()
        max_heap = MaxHeap(len(self.dictionary))
        for w in self.doc_vector:
            if (norm(self.doc_vector[w])!= 0):
                dist = self.calculate_sim(self.doc_vector[w])
                if (dist != 0):                             #index elemination
                    distances[w] = dist                                    #max heap construct
                    max_heap.insert(distances[w])
        chosen_docs = list()
        for i in range(self.K):
            x = max_heap.extractMax()
            chosen_doc = [key for (key, value) in distances.items() if value == x]
            print(chosen_doc)
            chosen_docs.append(chosen_doc[0])
            # import random
            # r = random.randrange(0, len(chosen_doc))
            # chosen_docs.append(r)
        # sorted_distance = sorted(distances.items(), key=operator.itemgetter(1),reverse=True)
        self.show_topK(chosen_docs)
        # print((sorted_distance))

    def vector_space(self):
        for i in range(0, self.totalDocs):
            for w in self.dictionary:
                if (i not in self.doc_vector):
                    self.doc_vector[i] = list()
                    if (int(self.index_tf[w][i]) != 0):
                        self.doc_vector[i].append(
                            (1 + math.log10(self.index_tf[w][i])) * math.log10(self.totalDocs / len(self.dictionary[w])))
                    else:
                        self.doc_vector[i].append(0)
                else :
                    if (self.index_tf[w][i] != 0):
                        self.doc_vector[i].append((1+ math.log10(self.index_tf[w][i]))* math.log10(self.totalDocs / len(self.dictionary[w])))
                    else:
                        self.doc_vector[i].append(0)

        # for w in (self.doc_vector):
        #      print(str(w) + " " + str(self.doc_vector[w]))
        return self.doc_vector
    def show_topK(self,chosen_docs):
        i = 0
        file = pandas.read_csv("ir-news-0-2.csv")
        for key in range(len(chosen_docs)):
            print("The " + str(i+1) + " Relevent Document")
            clean_text = BeautifulSoup(file['content'][chosen_docs[key]], "html.parser").getText()
            print(clean_text)
            if (i == self.K):
                break
            i =i+1
    def calculate_sim(self,doc):
        return  1 - spatial.distance.cosine(self.query_vector, doc)

# inv = inverted()
# dictionary,total ,index_tf= inv.create_second_inverted()
import time


query_processor = query_processor()
for i in range(10):
    query_processor.process()
    # query_processor.champion_process()
