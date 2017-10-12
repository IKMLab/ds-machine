import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionEncoder(nn.Module):

    """
    Encode a sequence of words into a vector
    """

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, batch_norm_out=True):
        super(ConvolutionEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embedding_size)) for ks in kernel_sizes])
        out_size = kernel_num * len(kernel_sizes)
        self.batch_norm_out = batch_norm_out

        if self.batch_norm_out:
            self.batch_norm = nn.BatchNorm1d(out_size)

    def forward(self, inputs):

        """
        :param inputs: a index sequence with shape : (N, T)
        :return: 
        """

        inputs_embedd = self.embedding(inputs)  # (N, T, D)
        inputs_embedd = inputs_embedd.unsqueeze(1)  # (N, 1, T, D)

        inputs_conv = [F.relu(conv(inputs_embedd)).squeeze(3) for conv in self.convs]  # [(N, K_num, W), ...]
        inputs_maxed = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs_conv]  # [(N, K_num), ...]
        out = torch.cat(inputs_maxed, 1)

        if self.batch_norm_out:
            out = self.batch_norm(out)

        return out
