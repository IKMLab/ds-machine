import torch
import torch.nn as nn

from trainer.utils.save import ModelManager
from trainer.utils.accuracy import BinaryAccuracyCalculator
from trainer.utils.tensorboard import Logger


class QATrainer(object):

    def __init__(self, data_transformer, model, use_cuda=True, checkpoint_path='checkpoint/CNNQA.pt'):
        self.data_transformer = data_transformer
        self.model = model
        self.model_manager = ModelManager(path=checkpoint_path)
        self.criterion = nn.NLLLoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.acc_calculator = BinaryAccuracyCalculator()
        self.logger = Logger('./logs')

        # for logging, would be replaced after connecting to tensorboard
        self.verbose_round = 1000
        self.save_round = 50000


    def train(self, epochs, batch_size=30, positive_sample_size=55, negative_sample_size=45):
        step = 0

        for i in range(epochs):
            epoch_loss = 0
            for(inputs, targets) in self.data_transformer.negative_batch(batch_size, positive_sample_size, negative_sample_size):
                query_var, answer_var = inputs
                logits, batch_loss = self._train_batch(query_var, answer_var, targets)
                epoch_loss += batch_loss.data[0]
                if (step + 1) % self.verbose_round == 0:
                    acc, pos_acc, neg_acc = self.acc_calculator.get_accuracy_torch(logits, targets, 0.375)
                    print("Epoch %d batch %d: Batch Loss:%.5f\tAcc:%.5f\tPos:%.5f\tNeg:%.5f"
                          % (i + 1, step + 1, batch_loss.data[0], acc, pos_acc, neg_acc))

                if (step + 1) % self.save_round == 0:
                    print(step, "Saving model")
                    self.model_manager.save_model(self.model)
                step += 1

                info = {
                    'batch_loss': batch_loss.data[0]
                }
                self.tensorboar_logging(info, step)


            print("Epoch %d total loss: %.5f" % (i, epoch_loss))

    def _train_batch(self, query_var, answer_var, target_var):
        # back-prop & optimize
        self.optimizer.zero_grad()
        logits = self.model.forward(query_var, answer_var)
        batch_loss = self.criterion(logits, target_var)
        batch_loss.backward()
        self.optimizer.step()

        return logits, batch_loss

    def tensorboar_logging(self, info, step):
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step)

    def load_pretrained_model(self, model_path):
        self.model_manager.load_model(self.model, model_path)


class SimTrainer(object):

    def __init__(self, data_transformer, model, use_cuda=True, checkpoint_path='checkpoint/CNNQA.pt'):
        self.data_transformer = data_transformer
        self.model = model
        self.model_manager = ModelManager(path=checkpoint_path)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.acc_calculator = BinaryAccuracyCalculator()
        self.logger = Logger('./logs')

        # for logging, would be replaced after connecting to tensorboard
        self.verbose_round = 1000
        self.save_round = 50000

    def cosine_sim_loss(self, pos_loss, neg_loss):
        loss = 0.1 - pos_loss + neg_loss
        return loss

    def train(self, epochs, batch_size=30, positive_sample_size=30, negative_sample_size=40):
        step = 0

        for i in range(epochs):
            epoch_loss = 0
            for query_pos_var, query_neg_var, answer_pos_var, answer_neg_var in self.data_transformer.regression_batch(batch_size, positive_sample_size, negative_sample_size):
                batch_loss = self._train_batch(query_pos_var, query_neg_var, answer_pos_var, answer_neg_var)
                epoch_loss += batch_loss.data[0]
                if (step + 1) % self.verbose_round == 0:
                    #acc, pos_acc, neg_acc = self.acc_calculator.get_accuracy_torch(logits, targets, 0.5)
                    print("Epoch %d batch %d: Batch Loss:%.5f\t"
                          % (i + 1, step + 1, batch_loss.data[0]))

                if (step + 1) % self.save_round == 0:
                    print(step, "Saving model")
                    self.model_manager.save_model(self.model)
                step += 1

                info = {
                    'batch_loss': batch_loss.data[0]
                }
                self.tensorboar_logging(info, step)


            print("Epoch %d total loss: %.5f" % (i, epoch_loss))

    def _train_batch(self, query_pos_var, query_neg_var, answer_pos_var, answer_neg_var):
        # back-prop & optimize
        self.optimizer.zero_grad()
        pos_sim = self.model.forward(query_pos_var, answer_pos_var)
        neg_sim = self.model.forward(query_neg_var, answer_neg_var)

        batch_loss = self.cosine_sim_loss(pos_sim, neg_sim)
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def tensorboar_logging(self, info, step):
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step)

    def load_pretrained_model(self, model_path):
        self.model_manager.load_model(self.model, model_path)