import torch
import random
import numpy as np

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
        sequence = [word for word in sequence]
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def batches(self, batch_size):
        pass
        # TODO

    def negative_batch(self, batch_size, positive_size, negative_size):
        # oversample the pos answer, undersample the neg answer
        # a batch is composed of #batch_size * negative_size pos answer and #batch_size * #nagative_size neg answers

        qa_pairs = list(zip(self.query_sequences, self.answer_sequences))
        random.shuffle(qa_pairs)
        query_sequences, answer_sequences = zip(*qa_pairs)

        pos_mini_batches = [
            (query_sequences[k: k + batch_size], answer_sequences[k: k + batch_size])
            for k in range(0, len(answer_sequences), batch_size)
        ]

        neg_mini_batches = [
            (query_sequences[k: k + batch_size] * negative_size,
             random.sample(answer_sequences, len(query_sequences[k: k + batch_size]) * negative_size))
            for k in range(0, len(answer_sequences), batch_size)
        ]

        for i in range(len(pos_mini_batches)):
            query_pos_batch, answer_pos_batch = pos_mini_batches[i]
            query_pos_batch = query_pos_batch * positive_size
            answer_pos_batch = answer_pos_batch * positive_size

            query_neg_batch, answer_neg_batch = neg_mini_batches[i]
            pos_targets = [1] * len(query_pos_batch)
            neg_targets = [0] * len(query_neg_batch)

            query_batch = query_pos_batch + query_neg_batch
            answer_batch = answer_pos_batch + tuple(answer_neg_batch)
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

    def regression_batch(self, batch_size, positive_size, negative_size):

        qa_pairs = list(zip(self.query_sequences, self.answer_sequences))
        random.shuffle(qa_pairs)
        query_sequences, answer_sequences = zip(*qa_pairs)

        pos_mini_batches = [
            (query_sequences[k: k + batch_size], answer_sequences[k: k + batch_size])
            for k in range(0, len(answer_sequences), batch_size)
        ]

        neg_mini_batches = [
            (query_sequences[k: k + batch_size] * negative_size,
             random.sample(answer_sequences, len(query_sequences[k: k + batch_size]) * negative_size))
            for k in range(0, len(answer_sequences), batch_size)
        ]

        for i in range(len(pos_mini_batches)):
            query_pos_batch, answer_pos_batch = pos_mini_batches[i]
            query_pos_batch = query_pos_batch * positive_size
            answer_pos_batch = answer_pos_batch * positive_size
            query_neg_batch, answer_neg_batch = neg_mini_batches[i]

            query_pos_batch_max_len = max([len(query) for query in query_pos_batch])
            query_neg_batch_max_len = max([len(query) for query in query_neg_batch])
            answer_pos_batch_max_len = max([len(answer) for answer in answer_pos_batch])
            answer_neg_batch_max_len = max([len(answer) for answer in answer_neg_batch])

            query_pos_padded = [self.pad_sequence(query, query_pos_batch_max_len) for query in query_pos_batch]
            query_neg_padded = [self.pad_sequence(query, query_neg_batch_max_len) for query in query_neg_batch]
            answer_pos_padded = [self.pad_sequence(answer, answer_pos_batch_max_len) for answer in answer_pos_batch]
            answer_neg_padded = [self.pad_sequence(answer, answer_neg_batch_max_len) for answer in answer_neg_batch]

            query_pos_var = Variable(torch.LongTensor(query_pos_padded))  # (B, T)
            query_neg_var = Variable(torch.LongTensor(query_neg_padded))  # (B, T)
            answer_pos_var = Variable(torch.LongTensor(answer_pos_padded))  # (B, T)
            answer_neg_var = Variable(torch.LongTensor(answer_neg_padded))  # (B, T)

            if self.use_cuda:
                query_pos_var = query_pos_var.cuda()
                query_neg_var = query_neg_var.cuda()
                answer_pos_var = answer_pos_var.cuda()
                answer_neg_var = answer_neg_var.cuda()

            yield query_pos_var, query_neg_var, answer_pos_var, answer_neg_var

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
