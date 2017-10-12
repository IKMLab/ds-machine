import numpy as np

class QASampler(object):

    def sample_negative_answers(self, answer_list, batch_size):
        """Return a list of negative answers"""
        return np.random.choice(answer_list, batch_size)

