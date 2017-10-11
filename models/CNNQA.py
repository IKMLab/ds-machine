import torch
import torch.nn as nn

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

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, hidden_size, out_size, conv_over_qa=False):
        super(ConvolutionDiscriminator, self).__init__()
        self.query_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.answer_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)

        if conv_over_qa:
            self.conv_onver_qa = True
            flatten_size = len(kernel_sizes) * kernel_num
            self.qa_convolution = nn.Conv2d(1, 32, kernel_sizes=(3, 2), padding=1) # 3 for same shape, 2 for (q,a)
        else:
            flatten_size = len(kernel_sizes) * kernel_num * 3

        self.h1 = nn.Linear(flatten_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)  # binary classification
        self.softmax = nn.LogSoftmax()

    def forward(self, query, answer, conv_over_conv=False):
        # D = len(kernel_sizes) * kernel_num
        q_vec = self.query_encoder.forward(query)  # (B, D)
        a_vec = self.answer_encoder.forward(answer)  # (B, D)

        if self.conv_onver_qa:
            qa_features = torch.cat((q_vec, a_vec), dim=1)  # (B, 2 * D)
            qa_features = qa_features.unsqueeze(1)  # (B, 1, 2 * D)
            qa_stacks = qa_features.view(qa_features.size(0), 2, -1)  # (B, 2, D)
            qa_stacks = qa_stacks.unsqueeze(1) # (B, 1, 2, D) -> an qa image with 1 channel and h,w = 2,D
            qa_conv = self.qa_convolution(qa_stacks) # (B, 32, 1, D)
            qa_features = qa_conv.squeeze(2).view(qa_conv.size(0), -1) # restore all features

        else:
            qa_prod = torch.bmm(q_vec, a_vec)  # (B, D)
            qa_features = torch.cat((q_vec, a_vec, qa_prod), dim=1)  # (B, 3 * D)

        h1 = self.h1(qa_features)
        logit = self.softmax(self.out(h1))
        return logit


