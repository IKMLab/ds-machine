import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from DSMachine.models.encoders import RNNEncoder, ConvEncoder
from DSMachine.configs import sentiment_config
from torch.nn.init import xavier_normal

class RNNSentimentClassifier(nn.Module):

    def __init__(self, vocab_size, hidden_size, output_size, layers):
        super(RNNSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = RNNEncoder.RNNEncoder(
            hidden_size=hidden_size,
            embedding_size=hidden_size,
            rnn_cell=sentiment_config.rnn_cell,
            bidirectional=sentiment_config.bidirectional,
            batch_first=sentiment_config.batch_first,
            embedding=self.embedding,
            num_layers=layers,
            dropout=sentiment_config.dropout
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, mean_on_time=True):
        input_vars, input_lengths = inputs
        output, hidden = self.encoder.forward(input_vars, input_lengths)

        if mean_on_time:
            output = torch.mean(output, dim=0)
        else:
            output = output[-1].squeeze(0) # squeeze the time dimension
        out = F.softmax(self.out(output))
        return out

class AttentionSentimentClassifier(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_size, layers):
        super(AttentionSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(embedding_size, hidden_size)
        self.encoder = RNNEncoder.RNNEncoder(
            hidden_size=hidden_size,
            rnn_cell=sentiment_config.rnn_cell,
            bidirectional=sentiment_config.bidirectional,
            batch_first=sentiment_config.batch_first,
            embedding=self.embedding,
            embedding_size = embedding_size,
            num_layers=layers,
            dropout=sentiment_config.dropout
        )

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, mean_on_time=True):
        input_vars, input_lengths = inputs
        output, hidden = self.encoder.forward(input_vars, input_lengths)

        if mean_on_time:
            output = torch.mean(output, dim=0)
        else:
            output = output[-1].squeeze(0) # squeeze the time dimension
        out = F.softmax(self.out(output))
        return out

class CNNSentimentClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, kernel_sizes, kernel_num):
        super(CNNSentimentClassifier, self).__init__()
        self.convolution_encoder = ConvEncoder.ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num, batch_norm_out=True)
        flatten_size = len(kernel_sizes) * kernel_num
        self.fc = nn.Linear(flatten_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def _weights_init(self, torch_module):
        xavier_normal(torch_module.weight.data, gain=math.sqrt(2.0))

    def forward(self, inputs, mean_on_time=False):
        inputs_var, _ = inputs
        inputs_var = inputs_var.transpose(0, 1)
        context = self.convolution_encoder(inputs_var)
        h1 = F.relu(self.fc(context))
        out = F.softmax(self.out(h1))
        return out

class SigmoidSentimentClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, kernel_sizes, kernel_num):
        super(SigmoidSentimentClassifier, self).__init__()
        self.convolution_encoder = ConvEncoder.ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num, batch_norm_out=True)
        flatten_size = len(kernel_sizes) * kernel_num
        self.fc = nn.Linear(flatten_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def _weights_init(self, torch_module):
        xavier_normal(torch_module.weight.data, gain=math.sqrt(2.0))

    def forward(self, inputs, mean_on_time=False):
        inputs_var, _ = inputs
        inputs_var = inputs_var.transpose(0, 1)
        context = self.convolution_encoder(inputs_var)
        h1 = F.relu(self.fc(context))
        out = F.sigmoid(self.out(h1))
        return out