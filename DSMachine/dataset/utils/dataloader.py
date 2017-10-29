from DSMachine.dataset.utils import voacb
from DSMachine.utils import log

class DataLoader(object):

    def __init__(self, dataset_path=None):
        self._data = []
        self.log = log.Log()
        self.vocab = voacb.Vocabulary()

        if dataset_path is not None:
            self._load_data(dataset_path)

    def load_data(self, dataset_path):
        self.log.info("[Data loader]: Loading the dataset at {}.".format(dataset_path))
        with open(dataset_path, 'r', encoding='utf-8') as dataset:
            for line in dataset:
                line = line.strip('\n')
                self._data.append(line)
        self.log.info("[Data loader]: the dataset has been loaded.")

    def _build_vocab(self):
        self.log.info("[Data loader]: Building the vocabulary.")
        self.vocab.build_vocab(self.data)

    @property
    def data(self):
        assert self._data is not None, "[Data loader]: Please load the dataset before using it."
        return self._data