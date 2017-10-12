import numpy as np

from trainer.utils.save import ModelManager
from dataset.data_helper import DataTransformer

class QAEvaluator(object):

    def __init__(self, model, dataset_path='dataset/Gossiping-QA-Dataset.txt', checkpoint_path='checkpoint/CNNQA.pt'):
        self.model_loader = ModelManager(checkpoint_path)
        self.data_transformer = DataTransformer(dataset_path, use_cuda=True, min_length=5)
        self.model = model
        self.model_loader.load_model(self.model)  # restore the weights of model

    def predict(self, sentence):
        all_prob = np.array([])
        for query_var, answer_var in self.data_transformer.evaluation_batch(sentence):
            logits = self.model.forward(query_var, answer_var)
            logits = logits.data.cpu().numpy()
            pos_prob = logits[:, 1]
            all_prob = np.concatenate((all_prob, pos_prob), axis=0)
        index = np.argmax(all_prob)
        print(self.data_transformer.vocab.answer_list[index])

    def interactive_mode(self):
        while True:
            user_input = input("Say something...")
            self.predict(user_input)
