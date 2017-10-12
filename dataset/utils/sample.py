import random
import numpy as np

class QASampler(object):

    def sample_negative_answers(self, answer_list, batch_size):
        """Return a list of negative answers"""
        return np.random.choice(answer_list, batch_size)

    def sample_negative_batch(self, query_list, answer_list, batch_size, negative_size):
        neg_ans_list = np.random.permutation(answer_list)


