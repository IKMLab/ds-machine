import torch
import torch.nn as nn

from trainer.utils.save import ModelManager
from trainer.utils.accuracy import BinaryAccuracyCalculator


class QATrainer(object):

    def __init__(self, data_transformer, model, use_cuda=True, checkpoint_path='checkpoint/CNNQA.pt'):
        self.data_transformer = data_transformer
        self.model = model
        self.model_manager = ModelManager(path=checkpoint_path)
        self.criterion = nn.NLLLoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.acc_calculator = BinaryAccuracyCalculator()

        # for logging, would be replaced after connecting to tensorboard
        self.verbose_round = 5000
        self.save_round = 50000

    def train(self, epochs, batch_size=20, negative_sample_size=10):
        for i in range(epochs):
            print("Epoch", i+1)
            for batch_id, (inputs, targets) in enumerate(self.data_transformer.negative_batch(batch_size, negative_sample_size)):
                query_var, answer_var = inputs
                batch_loss, accuracy = self._train_batch(query_var, answer_var, targets)

                if (batch_id + 1) % self.verbose_round == 0:
                    print("Epoch", i + 1, "batch", batch_id + 1, "Batch Loss", batch_loss.data[0], "Accuracy", accuracy)
                    # print("Positive accuracy", pos_acc)
                    # print("Negative accuracy", neg_acc)

                if (batch_id + 1) % self.save_round == 0:
                    print(batch_id, "Saving model")
                    self.model_manager.save_model(self.model)

    def _train_batch(self, query_var, answer_var, target_var):
        # back-prop & optimize
        self.optimizer.zero_grad()
        logits = self.model.forward(query_var, answer_var)
        batch_loss = self.criterion(logits, target_var)
        batch_loss.backward()
        self.optimizer.step()
        acc, pos_acc, neg_acc = self.acc_calculator.get_accuracy_torch(logits, target_var, 0.5)

        return batch_loss, acc, pos_acc, neg_acc