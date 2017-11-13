import opencc
import numpy as np

from DSMachine.dataset import data_helper
from DSMachine.trainer.utils import save


class QAEvaluator(object):

    def __init__(self, model, data_transformer, checkpoint_path):
        self.model_loader = save.ModelManager(checkpoint_path)
        self.data_transformer = data_transformer
        self.model = model
        self.model_loader.load_model(self.model)  # restore the weights of model

    def predict(self, sentence, topk=1):
        all_prob = np.array([])
        for query_var, answer_var in self.data_transformer.evaluation_batch(sentence):
            logits = self.model.forward(query_var, answer_var)
            logits = logits.data.cpu().numpy()
            pos_prob = logits[:, 1]
            all_prob = np.concatenate((all_prob, pos_prob), axis=0)
        indices = np.argpartition(all_prob, -topk)[-topk:]
        print(all_prob.shape)
        for index in indices:
            print(self.data_transformer.vocab.answer_list[index], all_prob[index])

    def interactive_mode(self, topk=20):
        while True:
            user_input = input("Say something...")
            self.predict(user_input, topk)

class SentimentEvaluator(object):

    def __init__(self, model, dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data', checkpoint_path='checkpoint/CNNQA.pt'):
        self.model_loader = save.ModelManager(checkpoint_path)
        self.data_transformer = data_helper.SentimentDataTransformer(dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data',
                                                    emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
                                                    use_cuda=True,
                                                    char_based=True)
        self.model = model
        self.model_loader.load_model(self.model)  # restore the weights of model

    def predict(self, sentence):
        sentence_var = self.data_transformer.evaluation_batch(sentence)
        logits = self.model.forward(sentence_var)
        predict_classes = logits.max(dim=1)[1]

        if predict_classes.data[0] == 0:
            return "Happy"
        else:
            return "Sad"

    def interactive_mode(self):
        while True:
            user_input = input("Say something...")
            user_input = self.simplify(user_input)
            print(self.predict(user_input))

    def simplify(self, text):
        return opencc.convert(text)

class PTTSentimentEvaluator(object):

    def __init__(self, model, data_transformer, checkpoint_path='checkpoint/clipped.pkt'):
        self.model_loader = save.ModelManager(checkpoint_path)
        self.data_transformer = data_transformer
        self.model = model
        self.model_loader.load_model(self.model)  # restore the weights of model

    def predict(self, sentence):
        sentence_var = self.data_transformer.evaluation_batch(sentence)
        print("Sentence" ,sentence_var)
        logits = self.model(sentence_var)
        print(logits)
        predict_classes = logits.max(dim=1)[1]

        if predict_classes.data[0] == 0:
            return "Happy"
        else:
            return "Sad"

    def interactive_mode(self):
        while True:
            user_input = input("Say something...")
            print(self.predict(user_input))