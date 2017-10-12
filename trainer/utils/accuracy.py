import numpy as np


class BinaryAccuracyCalculator(object):

    def get_accuracy_torch(self, logits, targets, pos_ratio=0):
        """
        :param logits: torch.Variable
        :param targets: torch.Variable
        :return:
        """
        predict_classes = logits.max(dim=1)[1]
        diff = predict_classes - targets
        false_prediction = targets.abs(diff).cpu().data.numpy()[0]
        accuracy = 1 - (false_prediction / diff.size(0))

        if pos_ratio == 0:
            return accuracy
        else:
            total_size = diff.size(0)
            pos_size = int(total_size * pos_ratio)
            neg_size = 1 - pos_size
            pos_false_prediction = false_prediction[:pos_size]
            neg_false_prediction = false_prediction[pos_size:]
            pos_accuracy = 1 - (pos_false_prediction / pos_size)
            neg_accuracy = 1 - (neg_false_prediction / neg_size)

            return accuracy, pos_accuracy, neg_accuracy
