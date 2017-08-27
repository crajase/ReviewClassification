import re

stop_words = []


def get_clean_wrd_list(line):
    line = re.sub("n't", " not", str(line))
    line = re.sub("[^a-zA-Z?!.@ ]", "", str(line))
    word_list = []
    for word in line.strip().split():
        word_lower = word.lower()
        if word_lower not in stop_words and len(word_lower) > 2:
            word_list.append(word_lower)
    return word_list
