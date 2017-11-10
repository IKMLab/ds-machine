import torch

class ModelManager(object):

    def __init__(self, path=None):
        self.path = path

    def save_model(self, model, path=None):
        path = self.path if path is None else path
        torch.save(model.state_dict(), path)
        print("Model has been saved as %s.\n" % path)

    def load_model(self, model, path=None):
        path = self.path if path is None else path
        model.load_state_dict(torch.load(path))
        model.eval()
        print("A pre-trained model at %s has been loaded.\n" % path)