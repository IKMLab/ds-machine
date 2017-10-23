import torch.nn as nn


class RNNEncoder(nn.Module):

    def __init__(self, rnn_cell, bidirectional, num_layers, hidden_size, dropout, embeddings, batch_first):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embeddings = embeddings
        self.batch_first = batch_first
        self.input_dropout = nn.Dropout(p=dropout)
        self.rnn = getattr(nn, rnn_cell)(
            input_size=embeddings.embedding_size,
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
