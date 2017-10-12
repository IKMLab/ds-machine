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
        self.verbose_round = 2000
        self.save_round = 50000

    def train(self, epochs, batch_size=30, negative_sample_size=30):
        for i in range(epochs):
            for batch_id, (inputs, targets) in enumerate(self.data_transformer.negative_batch(batch_size, negative_sample_size)):
                query_var, answer_var = inputs
                logits, batch_loss = self._train_batch(query_var, answer_var, targets)

                if (batch_id + 1) % self.verbose_round == 0:
                    acc, pos_acc, neg_acc = self.acc_calculator.get_accuracy_torch(logits, targets, 0.5)
                    print("Epoch %d batch %d: Batch Loss:%.5f\tAcc:%.5f\tPos:%.5f\tNeg:%.5f"
                          % (i + 1, batch_id + 1, batch_loss.data[0], acc, pos_acc, neg_acc))

                if (batch_id + 1) % self.save_round == 0:
                    print(batch_id, "Saving model")
                    self.model_manager.save_model(self.model, 'checkpoint/CNNQA.pt')

    def _train_batch(self, query_var, answer_var, target_var):
        # back-prop & optimize
        self.optimizer.zero_grad()
        logits = self.model.forward(query_var, answer_var)
        batch_loss = self.criterion(logits, target_var)
        batch_loss.backward()
        self.optimizer.step()

        return logits, batch_loss

    def load_pretrained_model(self, model_path):
        self.model_manager.load_model(self.model, model_path)