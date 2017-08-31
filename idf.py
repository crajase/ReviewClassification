import json
import math


class Idf(object):
    """docstring for ClassName"""
    def __init__(self, words_in_doc={}, N=0):
        if type(words_in_doc) is dict and type(N) is int:
            self.words_in_doc = words_in_doc
            self.N = N
        else:
            raise ValueError('Provide dict object and int object')

    def save(self, fileName):
        with open(fileName, 'w') as f:
            json.dump(self.__dict__, f)

    def build(self, sent_arr):
        for words in sent_arr:
            words_freq = self.uniq_wrd_cnt(words)
            self.__update_words_in_doc(words_freq)

    def clear(self):
        self.words_in_doc = {}
        self.N = 0

    def clear_and_build(self, sent_arr):
        self.clear()
        self.build(sent_arr)

    def uniq_wrd_cnt(self, words):
        words_freq = {}
        for word in words:
            if word not in words_freq:
                words_freq[word] = 1
            else:
                words_freq[word] += 1
        # self.update_words_in_doc(words_freq)
        return words_freq

    def get_idf_for(self, word):
        return self.words_in_doc[word] if word in self.words_in_doc else 0

    def calc_idf_for(self, word):
        return math.log(self.N / (1 + self.get_idf_for(word)))

    def calc_tf(self, words):
        return self.uniq_wrd_cnt(words)

    def calc_idf(self, words):
        return [self.calc_idf_for(word) for word in words]

    def calc_tf_idf(self, words):
        t_wrd_cnt = len(words)
        wrd_cnt = self.uniq_wrd_cnt(words)
        wrd_tf_idf = {}
        for wrd in wrd_cnt.keys():
            wrd_tf_idf[wrd] = (wrd_cnt[wrd] / t_wrd_cnt) * self.calc_idf_for(wrd)
        return wrd_tf_idf

    def __str__(self):
        ret = "Dict length = " + str(len(self.words_in_doc)) + "\n"
        ret += "Number of documents: " + str(self.N)
        return ret

    def __update_words_in_doc(self, words_freq):
        if(len(words_freq) > 0):
            self.N += 1
        for word in words_freq.keys():
            if word not in self.words_in_doc:
                self.words_in_doc[word] = 1
            else:
                self.words_in_doc[word] += 1

    @staticmethod
    def load(filename):
        idf_f = {}
        with open(filename) as data_file:
            idf_f = json.load(data_file)
        if 'words_in_doc' in idf_f and 'N' in idf_f:
            return Idf(idf_f['words_in_doc'], idf_f['N'])
        raise KeyError('The given json file doesnt have keys "N" or "words_in_doc"')
