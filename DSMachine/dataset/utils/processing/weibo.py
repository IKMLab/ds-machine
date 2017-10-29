import operator
import re

from collections import defaultdict

from DSMachine.dataset.utils.dataloader import DataLoader

class WeiboPreprocesser(DataLoader):

    def __init__(self, dataset_path=None):
        super(WeiboPreprocesser, self).__init__(dataset_path)
        self.q_list = []
        self.a_list = []
        self.tags = None

    def separate_question_answer(self):
        for idx, line in enumerate(self.data):
            if idx % 3 == 0:
                self.q_list.append(line)
            elif idx % 3 == 1:
                self.a_list.append(line)
        print(self.q_list[1:5])
        print(self.a_list[1:5])

    def clean_data(self):
        pass

    def parse_data(self):
        pass

    def extract_tags(self):
        self.log.info("[Extract Tag] Extracting possible emotes...")
        tags = defaultdict(int)
        for line in self.data:
            line = line.strip('\n')
            tag = ''
            append = False
            for i in range(len(line)):
                if line[i] == ']' and len(tag) != 0:
                    tags[tag] += 1
                    append = False
                    tag = ''
                elif append:
                    tag += line[i]
                elif line[i] == '[':
                    append = True
        self.log.info("[Extract Tag] Extracted successfully.")
        return tags

    def dump_tags(self, dump_file_name, lower_bound=100):
        if self.tags is None:
            tags = self.extract_tags()
        tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        with open(dump_file_name, 'w', encoding='utf-8') as dump:
            for tag, count in tags:
                if count >= lower_bound:
                    dump.write("{}\t{}\n".format(tag,count))
