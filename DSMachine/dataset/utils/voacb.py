import jieba

class Vocabulary(object):

    def __init__(self, char_based=True):
        self.word2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_words = 4
        self.max_length = 0
        self.char_based = char_based

    def build_vocab(self, sentences):
        for sentence in sentences:
            self.sentence_processing(sentence)

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as data:
            for line in data:
                word, idx = line.strip('\n').split(',')
                self.word2idx[word] = idx
                self.idx2word[id] = word
            self.num_words = len(self.word2idx)

    def dump_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as out:
            for word, idx in self.word2idx.items():
                out.write(word + ',' + idx + '\n')

    def sentence_processing(self, sentence):
        """Build the vocabulary based on input sentence"""
        if self.max_length < len(sentence):
            self.max_length = len(sentence)

        words = self.split_sequence(sentence)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = word
                self.num_words += 1

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.word2idx['SOS']] if add_sos else []

        for word in self.split_sequence(sequence):
            if word not in self.word2idx:
                index_sequence.append((self.word2idx['UNK']))
            else:
                index_sequence.append(self.word2idx[word])

        if add_eos:
            index_sequence.append(self.word2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for idx in indices:
            word = self.idx2word[idx]
            if word == "EOS":
                break
            else:
                sequence += word
        return sequence

    def split_sequence(self, sequence):
        if self.char_based:
            return [word for word in sequence]
        else:
            return [word for word in jieba.cut(sequence, cut_all=True)]

    def replace_with(self, vocab):
        pass

    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2word.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str

    def __len__(self):
        return len(self.word2idx)

class QAVocabulary(Vocabulary):

    """
    Processing the data with format:
    Q1\tA1
    Q2\tA2
    """

    def __init__(self):
        super(QAVocabulary, self).__init__()
        self.query_list = []
        self.answer_list = []

    def build_vocab_from_dataset(self, data_path, min_length):
        """Construct the relation between words and indices"""
        with open(data_path, 'r', encoding='utf-8') as dataset:
            for sentence in dataset:
                query, answer = sentence.strip('\n').split('\t')

                if len(query) > min_length and len(answer) > min_length:
                    self.query_list.append(query)
                    self.sentence_processing(query)
                    self.answer_list.append(answer)
                    self.sentence_processing(answer)