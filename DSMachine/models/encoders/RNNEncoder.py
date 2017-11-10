import torch
import torch.nn as nn


class RNNEncoder(nn.Module):

    def __init__(self, embedding, embedding_size, hidden_size, rnn_cell, bidirectional, num_layers, dropout, batch_first):
        super(RNNEncoder, self).__init__()

        self.bidirectional = bidirectional

        if self.bidirectional:
            hidden_size = hidden_size // 2

        self.embedding = embedding
        self.embedding_size = embedding_size
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(p=dropout)

        self.rnn = getattr(nn, rnn_cell)(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

    def forward(self, input_var, input_lengths, hidden=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=self.batch_first)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return output, hidden
