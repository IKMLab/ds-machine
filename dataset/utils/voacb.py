class Vocabulary(object):

    def __init__(self):
        self.word2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_words = 4
        self.max_length = 0
        self.query_list = []
        self.answer_list = []

    def build_vocab(self, data_path, min_length):
        """Construct the relation between words and indices"""
        with open(data_path, 'r', encoding='utf-8') as dataset:
            for sentence in dataset:
                query, answer = sentence.strip('\n').split('\t')

                if len(query) > min_length and len(answer) > min_length:
                    self.query_list.append(query)
                    self.sentence_processing(query)
                    self.answer_list.append(answer)
                    self.sentence_processing(answer)

    def sentence_processing(self, sentence):
        if self.max_length < len(sentence):
            self.max_length = len(sentence)

        words = self.split_sequence(sentence)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = words
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
            char = self.idx2word[idx]
            if char == "EOS":
                break
            else:
                sequence += char
        return sequence

    def split_sequence(self, sequence):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, a, b, e, t]
        """
        return [char for char in sequence]

    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2word.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str