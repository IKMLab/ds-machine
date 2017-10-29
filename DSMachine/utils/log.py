import logging


class Log(object):

    def __init__(self, level=logging.INFO):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)

    def info(self, msg):
        logging.info(msg)

    def warning(self, msg):
        logging.warning(msg)