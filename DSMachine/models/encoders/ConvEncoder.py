import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal


class ConvolutionEncoder(nn.Module):

    """
    Encode a sequence of words into a vector
    """

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, batch_norm_out=True, kmax_pooling=3):
        super(ConvolutionEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self._weights_init(self.embedding)  # xavier norm, TODO: change to a pre-trained word embedding

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embedding_size)) for ks in kernel_sizes])
        out_size = kernel_num * len(kernel_sizes)
        self.batch_norm_out = batch_norm_out

        if self.batch_norm_out:
            self.batch_norm = nn.BatchNorm1d(out_size)

    def _weights_init(self, torch_module):
        xavier_normal(torch_module.weight.data, gain=math.sqrt(2.0))

    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, inputs):
        """
        :param inputs: a index sequence with shape : (N, T)
        :return: 
        """
        inputs_embedded = self.embedding(inputs)  # (N, T, D)
        inputs_embedded = inputs_embedded.unsqueeze(1)  # (N, 1, T, D)

        inputs_conv = [F.relu(conv(inputs_embedded)).squeeze(3) for conv in self.convs]  # [(N, K_num, W), ...]
        inputs_maxed = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs_conv]  # [(N, K_num), ...]
        out = torch.cat(inputs_maxed, 1)

        if self.batch_norm_out:
            out = self.batch_norm(out)
        out = torch.nn.functional.tanh(out)
        return out


class KMaxConvolutionEncoder(nn.Module):

    """
    Encode a sequence of words into a vector
    """

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, batch_norm_out=True, kmax=3):
        super(ConvolutionEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self._weights_init(self.embedding)  # xavier norm, TODO: change to a pre-trained word embedding

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embedding_size)) for ks in kernel_sizes])
        out_size = kernel_num * len(kernel_sizes)
        self.batch_norm_out = batch_norm_out
        self.kmax = kmax

        if self.batch_norm_out:
            self.batch_norm = nn.BatchNorm1d(out_size)

    def _weights_init(self, torch_module):
        xavier_normal(torch_module.weight.data, gain=math.sqrt(2.0))

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, inputs):
        """
        :param inputs: a index sequence with shape : (N, T)
        :return: 
        """
        inputs_embedded = self.embedding(inputs)  # (N, T, D)
        inputs_embedded = inputs_embedded.unsqueeze(1)  # (N, 1, T, D)

        inputs_conv = [F.relu(conv(inputs_embedded)).squeeze(3) for conv in self.convs]  # [(N, K_num, W), ...]
        inputs_maxed = [self.kmax_pooling(i, 1, self.kmax) for i in inputs_conv]  # [(N, K_num, K), ...]
        inputs_maxed = inputs_maxed.views(inputs_maxed.size(0), inputs_maxed.size(1) * inputs_maxed(2), 1)
        inputs_maxed = inputs_maxed.squeeze(2)
        out = torch.cat(inputs_maxed, 1)

        if self.batch_norm_out:
            out = self.batch_norm(out)
        out = torch.nn.functional.tanh(out)
        return out
