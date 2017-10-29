import json

class Corpus(object):

    def __init__(self):
        self.corpus = []

    def load_corpus(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as dataset:
            for line in dataset:
                line = line.strip('\n')

    def add_sentence(self, sentence):
        self.corpus.append(set)


class CorpusGenerator(object):

    def __init__(self):
        self.corpus =


    def dump_corpus(self):
