import torch
import random

from torch.autograd import Variable

from dataset.utils.voacb import Vocabulary
from dataset.utils.sample import QASampler


class DataTransformer(object):

    def __init__(self, path, use_cuda=True, min_length=0):
        self.query_sequences = []
        self.answer_sequences = []
        self.use_cuda = use_cuda

        # Load and build the vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path, min_length)
        self.PAD_ID = self.vocab.word2idx["PAD"]
        self.SOS_ID = self.vocab.word2idx["SOS"]
        self.vocab_size = self.vocab.num_words
        self.max_length = self.vocab.max_length
        self.sampler = QASampler()

        self._build_training_set()

    def _build_training_set(self):
        for query in self.vocab.query_list:
            self.query_sequences.append(self.vocab.sequence_to_indices(query, add_eos=False))
        for answer in self.vocab.answer_list:
            self.answer_sequences.append(self.vocab.sequence_to_indices(answer, add_eos=False))


    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def batches(self, batch_size):
        # a batch is composed of 1 pos answer and #nagative_size neg answers
        # TODO Solve the unbalanced problem

        qa_pairs = list(zip(self.query_sequences, self.answer_sequences))
        random.shuffle(qa_pairs)
        query_sequences, answer_sequences = zip(*qa_pairs)

        pos_mini_batches = [
            (self.query_sequences[k: k + batch_size], self.answer_sequences[k: k + batch_size])
            for k in range(0, len(self.answer_sequences), batch_size)
        ]

        neg = self.answer_sequences[::-1]
        neg_mini_batches = [
            (self.query_sequences[k: k + batch_size], neg[k: k + batch_size])
            for k in range(0, len(self.answer_sequences), batch_size)
        ]

        for i in range(len(pos_mini_batches)):

            query_pos_batch, answer_pos_batch = pos_mini_batches[i]
            query_neg_batch, answer_neg_batch = neg_mini_batches[i]
            pos_targets = [1] * len(query_pos_batch)
            neg_targets = [0] * len(query_neg_batch)

            query_batch = query_pos_batch + query_neg_batch
            answer_batch = answer_pos_batch + answer_neg_batch
            targets = pos_targets + neg_targets

            query_batch_max_len = max([len(query) for query in query_batch])
            answer_batch_max_len = max([len(answer) for answer in answer_batch])

            query_padded = [self.pad_sequence(query, query_batch_max_len) for query in query_batch]
            answer_padded = [self.pad_sequence(answer, answer_batch_max_len) for answer in answer_batch]

            query_var = Variable(torch.LongTensor(query_padded))  # (B, T)
            answer_var = Variable(torch.LongTensor(answer_padded))  # (B, T)
            targets_var = Variable(torch.LongTensor(targets))

            if self.use_cuda:
                query_var = query_var.cuda()
                answer_var = answer_var.cuda()
                targets_var = targets_var.cuda()

            yield (query_var, answer_var), targets_var


    def evaluation_batch(self, query):

        eval_batch_size = 8192
        mini_batches = [
            self.answer_sequences[k: k + eval_batch_size]
            for k in range(0, len(self.answer_sequences), eval_batch_size)
        ]

        for answer_batch in mini_batches:

            query = self.vocab.sequence_to_indices(query)  # encode raw sentence to index seq
            query_batch = [query for _ in range(len(answer_batch))]

            query_batch_max_len = max([len(query) for query in query_batch])
            answer_batch_max_len = max([len(answer) for answer in answer_batch])

            query_padded = [self.pad_sequence(query, query_batch_max_len) for query in query_batch]
            answer_padded = [self.pad_sequence(answer, answer_batch_max_len) for answer in answer_batch]

            query_var = Variable(torch.LongTensor(query_padded))  # (B, T)
            answer_var = Variable(torch.LongTensor(answer_padded))  # (B, T)

            if self.use_cuda:
                query_var = query_var.cuda()
                answer_var = answer_var.cuda()

            yield query_var, answer_var
