import pickle
import logging
import os
import json
import codecs
import uuid
import time
import functools

from tools.tokenizer import Tokenizer
from pymorphy2 import MorphAnalyzer
from pyaspeller import YandexSpeller


class Utils:
    def __init__(self):
        self.log_path = os.path.join(os.path.dirname(__file__), '../')
        self.morph = MorphAnalyzer()
        self.speller = YandexSpeller()
        self.tokenizer = Tokenizer()

    def load_bin_data(self, path_to_data):
        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_binary(self, data, path):
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def load_json(self, file):
        file = os.path.join(file)
        sympt_tables = json.load(codecs.open(file, 'r', 'utf-8-sig'))
        return sympt_tables

    def save2text(self, path, data):
        with open(path, 'w') as outf:
            for el in data:
                outf.write(el + '\n')
        outf.close()

    def ngrams(self, massive, n):
        return [massive[pos: pos + n] for pos in range(len(massive) - n + 1)]

    def count_tokens(self, a_massive, b_massive):
        set_a = set(a_massive)
        set_b = set(b_massive)
        return {
            "len_a": len(set_a),
            "len_b": len(set_b),
            "len_union": len(set_a.union(set_b)),
            "len_inters": float(len(set_a.intersection(set_b)))
        }

    def init_logging(self, file_name):
        fmt = logging.Formatter('%(asctime)-15s %(message)s')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        log_dir_name = os.path.join(self.log_path, 'logs')
        log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '_%s.txt' % (file_name, )
        logging.info('Logging to {}'.format(log_file_name))
        logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

        return log_dir_name

    @functools.lru_cache(10000)
    def get_pos(self, s, index):
        parsed = self.morph.parse(s)[0]
        return {'pos': str(parsed.tag.POS), 'normal_form': parsed.normal_form, 'raw_token': s, 'index': index}

    def spell_checking(self, text):
        return {change['word']: change['s'][0] for change in self.speller.spell(text) if change['s']}

    def tokenization(self, text):
        return self.tokenizer.tokenize(text)
