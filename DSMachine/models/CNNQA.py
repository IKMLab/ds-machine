import torch
import torch.nn as nn

from DSMachine.models.encoders.ConvEncoder import ConvolutionEncoder


class ConvolutionCosineSimilarity(nn.Module):
    """
    Encode queries and answers to vectors and return their cosine similarity
    """

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num, with_linear=True):
        super(ConvolutionCosineSimilarity, self).__init__()
        self.conv_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-7)
        self.with_linear = with_linear
        if self.with_linear:
            pass
            #self.linear = nn.Linear()

    def forward(self, query, answer):
        q_vec = self.conv_encoder.forward(query)
        a_vec = self.conv_encoder.forward(answer)
        qa_sim = self.cosine_similarity(q_vec, a_vec)
        qa_sim_sum = torch.sum(qa_sim)
        return qa_sim_sum


class ConvolutionDiscriminator(nn.Module):

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num,
                 hidden_size, out_size, conv_over_qa=False, residual=False):
        super(ConvolutionDiscriminator, self).__init__()
        self.query_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.answer_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num)
        self.conv_over_qa = conv_over_qa
        self.residual = residual

        if self.conv_over_qa:
            out_channels = 32
            flatten_size = len(kernel_sizes) * kernel_num * out_channels
            self.qa_convolution = nn.Conv2d(1, out_channels, (2, 3), padding=(0, 1))  # 3 for same shape, 2 for (q,a)
        else:
            if self.residual:
                flatten_size = len(kernel_sizes) * kernel_num * 3
            else:
                flatten_size = len(kernel_sizes) * kernel_num

        self.fc = nn.Linear(flatten_size, hidden_size)  # maybe this layer should be removed before adding topk pooling
        self.out = nn.Linear(hidden_size, out_size)  # binary classification
        self.log_softmax = nn.LogSoftmax()

    def forward(self, query, answer):
        # D = len(kernel_sizes) * kernel_num
        q_vec = self.query_encoder.forward(query)  # (B, D)
        a_vec = self.answer_encoder.forward(answer)  # (B, D)

        if self.conv_over_qa:
            qa_features = torch.cat((q_vec, a_vec), dim=1)  # (B, 2 * D)
            qa_features = qa_features.unsqueeze(1)  # (B, 1, 2 * D)
            qa_stacks = qa_features.view(qa_features.size(0), 2, -1)  # (B, 2, D)
            qa_stacks = qa_stacks.unsqueeze(1)  # (B, 1, 2, D) -> an qa image with 1 channel and h,w = 2,D
            qa_conv = self.qa_convolution(qa_stacks)  # (B, 32, 1, D)
            qa_features = qa_conv.squeeze(2).view(qa_conv.size(0), -1)  # restore all features
        else:
            try:
                qa_correlation = q_vec * a_vec  # (B, D)
            except:
                print("Query size", query.size())
                print("Answer size", answer.size())
                print("q_vec size", q_vec.size())
                print("a_vec size", a_vec.size())

            if self.residual:
                qa_features = torch.cat((q_vec, a_vec, qa_correlation), dim=1)  # (B, 3 * D)
            else:
                qa_features = qa_correlation

        h1 = self.fc(qa_features)
        logits = self.log_softmax(self.out(h1))
        return logits


class GreedyClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, kernel_sizes, kernel_num,
                 hidden_size, out_size, kmax, residual=False):
        super(GreedyClassifier, self).__init__()
        self.query_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num, kmax_pooling=kmax)
        self.answer_encoder = ConvolutionEncoder(vocab_size, embedding_size, kernel_sizes, kernel_num, kmax_pooling=kmax)
        self.residual = residual

        if self.residual:
            flatten_size = len(kernel_sizes) * kmax * kernel_num * 3
        else:
            flatten_size = len(kernel_sizes) * kmax * kernel_num

        self.fc = nn.Linear(flatten_size, hidden_size)  # maybe this layer should be removed before adding topk pooling
        self.out = nn.Linear(hidden_size, out_size)  # binary classification
        self.log_softmax = nn.LogSoftmax()

    def forward(self, query, answer):
        # D = len(kernel_sizes) * kernel_num
        q_vec = self.query_encoder.forward(query)  # (B, D)
        a_vec = self.answer_encoder.forward(answer)  # (B, D)

        if self.conv_over_qa:
            qa_features = torch.cat((q_vec, a_vec), dim=1)  # (B, 2 * D)
            qa_features = qa_features.unsqueeze(1)  # (B, 1, 2 * D)
            qa_stacks = qa_features.view(qa_features.size(0), 2, -1)  # (B, 2, D)
            qa_stacks = qa_stacks.unsqueeze(1)  # (B, 1, 2, D) -> an qa image with 1 channel and h,w = 2,D
            qa_conv = self.qa_convolution(qa_stacks)  # (B, 32, 1, D)
            qa_features = qa_conv.squeeze(2).view(qa_conv.size(0), -1)  # restore all features
        else:
            try:
                qa_correlation = q_vec * a_vec  # (B, D)
            except:
                print("Query size", query.size())
                print("Answer size", answer.size())
                print("q_vec size", q_vec.size())
                print("a_vec size", a_vec.size())

            if self.residual:
                qa_features = torch.cat((q_vec, a_vec, qa_correlation), dim=1)  # (B, 3 * D)
            else:
                qa_features = qa_correlation

        h1 = self.fc(qa_features)
        logits = self.log_softmax(self.out(h1))
        return logits


