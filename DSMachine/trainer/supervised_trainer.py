import torch
import torch.nn as nn
from DSMachine.trainer.utils.accuracy import BinaryAccuracyCalculator
from DSMachine.trainer.utils.save import ModelManager
from DSMachine.trainer.utils.tensorboard import Logger
from DSMachine.evaluation.evaluator import SentimentEvaluator
from DSMachine.sentiment.sentiment_classify import CNNSentimentClassifier

"""
TODO REFACTOR
"""

class Trainer(object):

    def __init__(self, model, data_transformer, checkpoint_path):
        self.model_manager = ModelManager()
        self.logger = Logger('./logs')
        self.model = model
        self.data_transformer = data_transformer
        self.checkpoint_path = checkpoint_path

    def tensorboard_logging(self, info, step):
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step)

    def load_pretrained_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            self.model_manager.load_model(self.model, self.checkpoint_path)
        else:
            self.model_manager.load_model(self.model, checkpoint_path)

class ClassifierTrainer(Trainer):

    def __init__(self, data_transformer, model, learning_rate=0.001, checkpoint_path='checkpoint/SentimentClassifier.pt', loss=None):
        super(ClassifierTrainer, self).__init__(model, data_transformer, checkpoint_path)
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if loss is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = loss

        self.acc_calculator = BinaryAccuracyCalculator()
        self.logger = Logger('./logs')

        # for logging, would be replaced after connecting to tensorboard
        self.verbose_round = 100
        self.evaluation_round = 10000
        self.save_round = 10000

    def train(self, epochs, batch_size=128, pretrained=False, binary=False, sigmoid=False):
        step = 0

        if pretrained:
            self.model_manager.load_model(self.model, self.checkpoint_path)

        for i in range(epochs):
            epoch_loss = 0

            if binary:
                data_batch = self.data_transformer.binary_batch
            else:
                data_batch = self.data_transformer.balanced_batch

            for sentences_var, targets_var in data_batch(batch_size):
                batch_loss, accuracy, logits = self._train_batch(sentences_var, targets_var, sigmoid)
                epoch_loss += batch_loss.data[0]
                if (step + 1) % self.verbose_round == 0:
                    print("Epoch %d batch %d: Batch Loss:%.5f\t Accuracy:%.5f"
                          % (i + 1, step + 1, batch_loss.data[0], accuracy))

                if (step + 1) % self.save_round == 0:
                    print(step, "Saving model")
                    self.model_manager.save_model(self.model, path=self.checkpoint_path)
                step += 1

                if (step + 1) % self.evaluation_round == 0:
                    print(step, "Evaluation")
                    (input_var, input_lengths), output_var, seqs = self.data_transformer.evaluation_batch()
                    seqs = seqs.tolist()
                    logits = self.model.forward((input_var, input_lengths))
                    print("Logits", logits)
                    predict_classes = logits.data.cpu().numpy()
                    for s, t in zip(seqs, predict_classes):
                        print("Sentence", self.data_transformer.vocab.indices_to_sequence(s[0]))
                        print("Ground Truth", s[1], "Predict", t.data[0], t.data[1])

                info = {
                    'batch_loss': batch_loss.data[0]
                }
                self.tensorboard_logging(info, step)
            print("Epoch %d total loss: %.5f" % (i + 1, epoch_loss))

    def predict(self, sentence):
        sentence_var = self.data_transformer.evaluation_batch(sentence)
        logits = self.model.forward(sentence_var)
        predict_classes = logits.max(dim=1)[1]

        if predict_classes.data[0] == 0:
            return "Happy"
        else:
            return "Sad"

    def _train_batch(self, sentences_var, targets_var, sigmoid):
        # back-prop & optimize
        logits = self.model.forward(sentences_var)
        batch_loss = self.criterion(logits, targets_var)
        self.optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 10)
        self.optimizer.step()
        accuracy = self.acc_calculator.get_accuracy(logits, targets_var, sigmoid)
        return batch_loss, accuracy, logits

    def cv_train(self, k, epoch, batch_size=512, sigmoid=False, cv_checkpoint='checkpoint/cv_checkpoint'):
        fold_acc = 0.0
        fold = 0

        for x_train, y_train, x_test, y_test in self.data_transformer.k_fold(k):
            step = 0
            self.model_manager.save_model(self.model, cv_checkpoint)
            for i in range(epoch):
                epoch_loss = 0
                mini_batch = [
                    (x_train[k:k + batch_size], y_train[k:k + batch_size])
                    for k in range(0, len(x_train), batch_size)
                ]
                # train with k-1 fold
                for bx, by in mini_batch:
                    sentences_var, targets_var = self.data_transformer.batch(bx, by)
                    batch_loss, accuracy, logits = self._train_batch(sentences_var, targets_var, sigmoid=sigmoid)

                    epoch_loss += batch_loss.data[0]
                    if (step + 1) % self.verbose_round == 0:
                        print("Epoch %d with fold %d on step %d: Batch Loss:%.5f\t Accuracy:%.5f"
                              % (i + 1, fold + 1, step + 1, batch_loss.data[0], accuracy))

                    if (step + 1) % self.save_round == 0:
                        print(step, "Saving model")
                        self.model_manager.save_model(self.model, path=self.checkpoint_path)

                    if (step + 1) % self.evaluation_round == 0:
                        print(step, "Evaluation")
                        (input_var, input_lengths), output_var, seqs = self.data_transformer.evaluation_batch()
                        seqs = seqs.tolist()
                        logits = self.model.forward((input_var, input_lengths))
                        predict_classes = logits.data.cpu().numpy()
                        for s, t in zip(seqs, predict_classes):
                            print("Sentence", self.data_transformer.vocab.indices_to_sequence(s[0]))
                            print("Ground Truth", s[1], "Predict", t.data[0], t.data[1])
                    step += 1

                # evaluate on 1 fold
            sentences_var, output_var = self.data_transformer.batch(x_test, y_test)
            logits = self.model.forward(sentences_var)
            accuracy = self.acc_calculator.get_accuracy(logits, output_var, sigmoid)

            print("Fold %d, Accuracy %.5f" % (fold, accuracy))
            fold_acc += accuracy
            fold += 1

            self.model_manager.load_model(self.model, cv_checkpoint)

        print("Fold %d, with %d Epoch Accuracy: %.5f" % (fold + 1, epoch, fold_acc / k))
