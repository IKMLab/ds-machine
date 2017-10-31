import random
import torch
import numpy as np

from torch.autograd import Variable

from DSMachine.dataset.utils.voacb import QAVocabulary
from DSMachine.dataset.utils.dataloader import DataLoader
from DSMachine.dataset.utils.sample import QASampler

class SentimentDataTransformer(object):

    def __init__(self, dataset_path, emote_meta_path, use_cuda, char_based):
        self.use_cuda = use_cuda

        self.data_loader = DataLoader(char_based=char_based)
        self.data_loader.load_data(dataset_path=dataset_path)
        self.PAD_ID = self.data_loader.vocab.word2idx["PAD"]
        self.sentences = []
        self.tags = []

        self.emote_meta = {}
        self._load_emote_meta(emote_meta_path)
        self._build_training_set()
        self.vocab_size = self.data_loader.vocab.num_words

    def _build_training_set(self):
        for data in self.data_loader.data:
            tag, sentence = data.split(' ')
            sequences = self.data_loader.vocab.sequence_to_indices(sentence)
            self.sentences.append(sequences)
            self.tags.append(self.emote_meta[tag])
            #print(sequences, self.emote_meta[tag])

    def _load_emote_meta(self, emote_meta_path):
        res = DataLoader(char_based=False, dataset_path=emote_meta_path)
        for data in res.data:
            emote, id = data.split(' ')
            self.emote_meta[emote] = int(id)

    def mini_batches(self, batch_size):

        training_data = list(zip(self.sentences, self.tags))
        random.shuffle(training_data)
        sentences, tags = zip(*training_data)

        mini_batches = [
            (sentences[k:k+batch_size], tags[k:k+batch_size])
            for k in range(0, len(sentences), batch_size)
        ]

        for sentences, tags in mini_batches:
            input_seqs = sorted(sentences, key=lambda s: len(s), reverse=True)
            input_lengths = [len(s) for s in input_seqs]
            in_max = max(input_lengths)
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]
            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
            output_var = Variable(torch.LongTensor(tags))

            if self.use_cuda:
                input_var = input_var.cuda()
                output_var = output_var.cuda()

            yield (input_var, input_lengths), output_var

    def multi_mini_batches(self, batch_size):

        training_data = list(zip(self.sentences, self.tags))
        random.shuffle(training_data)
        input_seqs = sorted(training_data, key=lambda s: len(s[0]), reverse=True)
        sentences, tags = zip(*input_seqs)
        input_lengths = [len(s) for s in sentences]
        
        temp, result = [input_seqs[0]], []
        for i in range(1, len(input_seqs)):
            if input_lengths[i] == input_lengths[i-1]:
                temp.append(input_seqs[i])
            else:
                result.append(temp)
                temp = [input_seqs[i]]
        result.append(temp)

        for cand_batch in result:
            if len(cand_batch) < batch_size:
                sentence, tag = zip(*cand_batch)
                input_var = Variable(torch.LongTensor(list(sentence))).transpose(0, 1)
                output_var = Variable(torch.LongTensor(list(tag)))

                if self.use_cuda:
                    input_var = input_var.cuda()
                    output_var = output_var.cuda()

                yield input_var, output_var

            else:
                num_batch = int(len(cand_batch) / batch_size)
                cand_batch = cand_batch[: num_batch * batch_size]
                mini_batches = [
                    cand_batch[k: k + batch_size]
                    for k in range(0, len(cand_batch), batch_size)
                ]
                for batch in mini_batches:
                    sentence, tag = zip(*batch)
                    input_var = Variable(torch.LongTensor(list(sentence))).transpose(0, 1)
                    output_var = Variable(torch.LongTensor(list(tag)))

                    if self.use_cuda:
                        input_var = input_var.cuda()
                        output_var = output_var.cuda()

                    yield (input_var, len(sentence[0])), output_var


    def pad_sequence(self, sequence, max_length):
        sequence = [word for word in sequence]
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence


class DataTransformer(object):

    def __init__(self, path, use_cuda=True, min_length=0):
        self.query_sequences = []
        self.answer_sequences = []
        self.use_cuda = use_cuda

        # Load and build the vocab
        self.vocab = QAVocabulary()
        self.vocab.build_vocab_from_dataset(path, min_length)
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
