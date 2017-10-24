import numpy as np

from DSMachine.dataset import data_helper
from DSMachine.trainer.utils import save


class QAEvaluator(object):

    def __init__(self, model, dataset_path='dataset/Gossiping-QA-Dataset.txt', checkpoint_path='checkpoint/CNNQA.pt'):
        self.model_loader = save.ModelManager(checkpoint_path)
        self.data_transformer = data_helper.DataTransformer(dataset_path, use_cuda=True, min_length=5)
        self.model = model
        self.model_loader.load_model(self.model)  # restore the weights of model

    def predict(self, sentence, topk=1):
        all_prob = np.array([])
        for query_var, answer_var in self.data_transformer.evaluation_batch(sentence):
            logits = self.model.forward(query_var, answer_var)
            logits = logits.data.cpu().numpy()
            pos_prob = logits[:, 1]
            print(pos_prob.shape)
            all_prob = np.concatenate((all_prob, pos_prob), axis=0)
        indices = np.argpartition(all_prob, -topk)[-topk:]
        print(all_prob.shape)
        for index in indices:
            print(self.data_transformer.vocab.answer_list[index])

    def interactive_mode(self, topk=1):
        while True:
            user_input = input("Say something...")
            self.predict(user_input, topk)
