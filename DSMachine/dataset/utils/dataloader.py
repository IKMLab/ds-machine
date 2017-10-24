class DataLoader(object):

    def __init__(self, dataset_path=None):
        self._data = None

        if dataset_path is not None:
            self._loda_data(dataset_path)

    def _loda_data(self, dataset_path):
        print("[Data loader]: Loading the dataset at {}.".format(dataset_path))
        with open(dataset_path, 'w', encoding='utf-8') as dataset:
            for line in dataset:
                line = line.strip('\n')
                self._data.append(line)
        print("[Data loader]: the dataset has been loaded.")

    @property
    def data(self):
        assert self._data is not None, "[Data loader]: Please load the dataset before using it."
        return self._data


class WeiboDataLoader(DataLoader):

    def __init__(self, dataset_path=None):
        super(WeiboDataLoader, self).__init__(dataset_path)
        self.q_list = []
        self.a_list = []

    def separate_question_answer(self):
        for idx, line in enumerate(self.data):
            if idx % 3 == 0:
                self.q_list.append(line)
            elif idx % 3 == 1:
                self.a_list.append(line)

    def clean_data(self):
        pass

    def parse_data(self):
        pass

    def extract_tags(self):
        pass

    def dumps_data(self):
        pass
