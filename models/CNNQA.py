import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.ConvEncoder import ConvolutionEncoder


class ConvolutionCosineSimilarity(nn.Module):
    """
    Encoder queries and answers to vectors and return their cosine similarity
    """

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num):
        super(ConvolutionCosineSimilarity, self).__init__()
        self.query_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.answer_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-7)


    def forward(self, query, answer):
        q_vec = self.query_encoder.forward(query)
        a_vec = self.answer_encoder.forward(answer)
        qa_sim = self.cosine_similarity(q_vec, a_vec)
        return qa_sim


class ConvolutionDiscriminator(nn.Module):

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, hidden_size, out_size):
        super(ConvolutionDiscriminator, self).__init__()
        self.query_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.answer_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.h1 = nn.Linear(len(kernel_sizes) * kernel_num * 3, hidden_size)  # binary classification
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, query, answer):
        # D = len(kernel_sizes) * kernel_num
        q_vec = self.query_encoder.forward(query)  # (B, D)
        a_vec = self.answer_encoder.forward(answer)  # (B, D)
        qa_prod = torch.bmm(q_vec, a_vec)  # (B, D)
        qa_features = torch.cat((q_vec, a_vec, qa_prod), dim=1)  # (B, 3 * D)
        h1 = self.h1(qa_features)
        logit = self.softmax(self.out(h1))
        return logit


