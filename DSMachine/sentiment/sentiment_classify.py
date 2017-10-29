import torch
import torch.nn as nn

from DSMachine.models.encoders import RNNEncoder, ConvEncoder
from DSMachine.configs import sentiment_config

class RNNSentimentClassifier(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_size, layers):
        super(RNNSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(embedding_size, hidden_size)
        self.encoder = RNNEncoder.RNNEncoder(
            hidden_size=hidden_size,
            rnn_cell=sentiment_config.rnn_cell,
            bidirectional=sentiment_config.bidirectional,
            batch_first=sentiment_config.batch_first,
            embedding=self.embedding,
            num_layers=layers,
            dropout=sentiment_config.dropout
        )

        self.out = nn.Linear(hidden_size, output_size)
        self.out_gate = nn.LogSoftmax()

    def forward(self, inputs, mean_on_time=False):
        input_vars, input_lengths = inputs
        output, hidden = self.encoder.forward(input_vars, input_lengths)

        if mean_on_time:
            output = torch.mean(output, dim=0)
        else:
            output = output[-1].squeeze(0) # squeeze the time dimension
        out = self.out_gate(self.out(output))
        return out

class CNNSentimentClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, kernel_sizes, kernel_num):
        super(CNNSentimentClassifier, self).__init__()
        self.conv_encoder = ConvEncoder.ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        flatten_size = len(kernel_sizes) * kernel_num
        self.fc = nn.Linear(flatten_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.out_gate = nn.LogSoftmax()

    def forward(self, inputs, mean_on_time=False):
        inputs_var, _ = inputs
        inputs_var = inputs_var.transpose(0, 1)
        context = self.conv_encoder.forward(inputs_var)
        h1 = self.fc(context)
        out = self.out_gate(self.out(h1))
        return out

class GreedyHAN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, kernel_sizes, kernel_num):
        super(GreedyHAN, self).__init__()
        self.pos_neg_encoder = ConvEncoder.KMaxConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.neg_encoder = ConvEncoder.KMaxConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        flatten_size = len(kernel_sizes) * kernel_num * 3
        self.h1 = nn.Linear(flatten_size, hidden_size)
        self.pos_neg_out = nn.Linear(hidden_size, 2)

        self.h2 = nn.Linear(flatten_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.out_gate = nn.LogSoftmax()

    def forward(self, inputs, mean_on_time=False):
        inputs_var, _ = inputs
        inputs_var = inputs_var.transpose(0, 1)

        context = self.pos_neg_encoder.forward(inputs_var)

        # check is pos or neg
        h1 = self.h1(context)
        pos_or_neg = self.out_gate(self.pos_neg_out(h1))

        h2 = self.h2(context)
        out = self.out_gate(self.out(h2))
        return pos_or_neg, out

class SentimentAttentionClassifier(nn.Module):
    # TODO
    pass