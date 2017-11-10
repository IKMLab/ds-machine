import torch
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
        false_prediction = torch.abs(diff).cpu().data.numpy()
        accuracy = 1 - (false_prediction.sum() / diff.size(0))

        if pos_ratio == 0:
            return accuracy
        else:
            total_size = diff.size(0)
            pos_size = int(total_size * pos_ratio)
            neg_size = total_size - pos_size
            pos_false_prediction = false_prediction[:pos_size].sum()
            neg_false_prediction = false_prediction[pos_size:].sum()
            pos_accuracy = 1 - (pos_false_prediction / pos_size)
            neg_accuracy = 1 - (neg_false_prediction / neg_size)

            return accuracy, pos_accuracy, neg_accuracy

    def get_accuracy(self, logits, targets, sigmoid=False):


        if sigmoid:
            # [512, 1]
            logits = torch.round(logits)

            res = torch.eq(logits.view(-1).data, targets.view(-1).data)
            res = torch.sum(res)

            return res / targets.size(0)
        else:
            predict_classes = logits.max(dim=1)[1]
            corrections = torch.eq(predict_classes.view(-1).data, targets.view(-1).data)
            accuracy = torch.sum(corrections) / corrections.size(0)
            return accuracy
