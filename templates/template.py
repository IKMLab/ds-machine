import json, os


class Template(object):

    def __init__(self, name, slots, replies):
        self.meta = {}
        self.meta["name"] = name
        self.meta["slots"] = slots
        self.meta["replies"] = replies

    def serialize(self):
        return json.dumps(self.meta)


class TemplateHelper(object):

    def __init__(self):
        self.templates = []

    def load_templates(self, dir_path):
        for filename in os.listdir(dir_path):
            with open(filename, 'r', encoding='utf-8') as data:
                template = json.load(data)
                self.templates.append(template)

    def save_templates(self, dir_path):
        for template in self.templates:
            filename = template.meta["name"] + ".json"
            with open(filename, 'w', encoding='utf-8') as out:
                out.write(template.serialize())
