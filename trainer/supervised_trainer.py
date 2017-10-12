import torch
import torch.nn as nn
import numpy as np

from dataset.data_helper import DataTransformer
from trainer.utils.save import ModelManager


class QATrainer(object):

    def __init__(self, data_transformer, model, use_cuda=True, checkpoint_path='checkpoint/CNNQA.pt'):
        self.data_transformer = data_transformer
        self.model = model
        self.model_manager = ModelManager(path=checkpoint_path)
        self.criterion = nn.NLLLoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # for logging, would be replaced after connecting to tensorboard
        self.verbose_round = 500
        self.save_round = 5000

    def train(self, epochs, batch_size=256):
        for i in range(epochs):
            print("Epoch", i+1)
            for batch_id, (inputs, targets) in enumerate(self.data_transformer.batches(batch_size)):
                query_var, answer_var = inputs
                batch_loss, accuracy = self._train_batch(query_var, answer_var, targets)

                if batch_id % self.verbose_round == 0:
                    print(batch_id, "Batch Loss", batch_loss.data)
                    print(batch_id, "Accuracy", accuracy)
                if batch_id % self.save_round == 0:
                    print(batch_id, "Saving model")
                    self.model_manager.save_model(self.model)

    def _train_batch(self, query_var, answer_var, target_var):
        # back-prop & optimize
        self.optimizer.zero_grad()
        logits = self.model.forward(query_var, answer_var)
        batch_loss = self.criterion(logits, target_var)
        batch_loss.backward()

        # calcluate accuracy
        predict_classes = logits.max(dim=1)[1]
        diff = predict_classes - target_var
        false_predection = torch.abs(diff).sum()
        accuracy = 1 - (false_predection.cpu().data.numpy()[0] / diff.size(0))
        self.optimizer.step()

        return batch_loss, accuracy