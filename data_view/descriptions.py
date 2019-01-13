import os
import config
import functools
import logging
from collections import Counter

import pandas as pd
import random
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pyaspeller import YandexSpeller
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

from tools.utils import Utils
from tools.data_loader import DataObject, DataLoader
from tools.symptoms_vectorizer import SymptomsVectorizer
from config.Config_worker import Config
from tools.tokenizer import Tokenizer


class DescriptionsViewer:
    def __init__(self, plot_path, config_data):
        self.utils = Utils()
        self.tokenizer = Tokenizer(config_data=config_data)

        self.speller = YandexSpeller()
        self.morph = MorphAnalyzer()
        self.stopWords = set(stopwords.words('russian') + ['это', 'такои', 'какои', 'также'])

        self.plot_path = plot_path
        self.utils.init_logging('data_view_descriptions')
        self.max_ngrams_size = config_data.get("max_ngrams_size")

    def preprocessed(self, description):
        correction = self.spell_checking(description)
        tokens = self.tokenization(description)
        if correction:
            tokens = [correction.get(token, token) for token in tokens]
        normal_forms = [self.get_pos(tokens[index]) for index in range(len(tokens))]
        return normal_forms

    @functools.lru_cache(10000)
    def get_pos(self, s):
        parsed = self.morph.parse(s)[0]
        return parsed.normal_form

    def spell_checking(self, text):
        return {change['word']: change['s'][0] for change in self.speller.spell(text) if change['s']}

    def tokenization(self, text):
        return self.tokenizer.tokenize(text)

    def plot_frequency_distribution_of_ngrams(self,
                                              descr,
                                              ngram_range=(1, 2),
                                              num_ngrams=50):

        descr = [' '.join(descr[index]) for index in tqdm(range(len(descr)))]

        kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',
            'stop_words': self.stopWords
        }
        vectorizer = CountVectorizer(**kwargs)
        vectorized_texts = vectorizer.fit_transform(descr)

        all_ngrams = list(vectorizer.get_feature_names())
        num_ngrams = min(num_ngrams, len(all_ngrams))

        all_counts = vectorized_texts.sum(axis=0).tolist()[0]
        all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
            zip(all_counts, all_ngrams), reverse=True)])

        freq_distr_ngrams = pd.DataFrame()
        freq_distr_ngrams['ngrams'] = list(all_ngrams)[:num_ngrams]
        freq_distr_ngrams['counts'] = list(all_counts)[:num_ngrams]

        plt.rcParams['figure.figsize'] = 22, 8
        freq_distr_ngrams.plot(x='ngrams', y='counts', kind='barh')
        plt.savefig(os.path.join(self.plot_path, 'freq_distr_ngrams_range=%s.jpeg' % (str(ngram_range), )))

    def get_num_words_per_sample(self, descr):
        num_words = []
        for index in range(len(descr)):
            num_words.append(len(descr[index]))
        logging.info('descr_median_len: ' + str(np.median(num_words)))
        logging.info('descr_max_len: ' + str(np.max(num_words)))
        logging.info('descr_min_len: ' + str(np.min(num_words)))

        plt.hist(num_words, 60)
        plt.xlabel('Length of a sample')
        plt.ylabel('Number of samples')
        plt.title('Sample length distribution')
        plt.savefig(os.path.join(self.plot_path, 'sample_len_distr.jpeg'))

    def intersections_ngrams(self, descr, sympt, sympt_index, ngram_order, sample_size):
        result = []
        if ngram_order > 0:
            all_sympt = {s: ['_'.join(ngr) for ngr in self.utils.ngrams(s, ngram_order)]
                         for s in sympt_index}
        else:
            all_sympt = {s: [t for t in s if len(t) > 1] for s in sympt_index}
        for index in range(len(descr)):
            if ngram_order > 0:
                descr_prep = ['_'.join(ngr) for ngr in self.utils.ngrams(descr[index], ngram_order)]
            else:
                descr_prep = [t for t in self.preprocessed(descr[index]) if len(t) > 1]
            sympt_prep = [all_sympt[s] for s in sympt[index]]
            other_symptoms = [all_sympt[s] for s in all_sympt if s not in sympt[index]]
            _ = [len(set(descr_prep).intersection(set(s))) for s in sympt_prep]
            _ = [len(set(descr_prep).intersection(set(s))) for s in other_symptoms]

            intersections_s = [s for s in sympt_prep if set(descr_prep).intersection(set(s))]
            intersections_other_s = [s for s in other_symptoms if set(descr_prep).intersection(set(s))]
            result.append([len(intersections_s), len(intersections_other_s)])

        result = random.sample(result, sample_size)
        group_labels = [i for i in range(len(result))]
        group_metrics = ["aim_sympt", "other_sympt"]
        data = pd.DataFrame(result, index=group_labels, columns=group_metrics)
        data.plot.bar()
        plt.savefig(os.path.join(self.plot_path, 'inter_aim_vs_other_ngram_order=%s.jpeg' % (ngram_order,)))


if __name__ == '__main__':
    experiment_config = Config(file_name=config.CONFIG_PATH)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/')
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/plots/')
    tables_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/Tables/')

    data_loader = DataLoader(config_data=experiment_config, data_path=data_path)
    descr_viewer = DescriptionsViewer(plot_path=plot_path, config_data=experiment_config)
    sympt_vectorizer = SymptomsVectorizer(config_data=experiment_config, data_path=tables_path)
    data = data_loader.load_chunks()

    description_train = [descr_viewer.preprocessed(el.description) for el in tqdm(data)]
    # symptoms = [[descr_viewer.preprocessed(s) for s in sorted(el.symptoms)] for el in data]

    descr_viewer.get_num_words_per_sample(description_train)
    descr_viewer.plot_frequency_distribution_of_ngrams(description_train, ngram_range=(1, 1), num_ngrams=50)
    descr_viewer.plot_frequency_distribution_of_ngrams(description_train, ngram_range=(2, 2), num_ngrams=50)
    descr_viewer.plot_frequency_distribution_of_ngrams(description_train, ngram_range=(3, 3), num_ngrams=50)
    descr_viewer.plot_frequency_distribution_of_ngrams(description_train, ngram_range=(4, 4), num_ngrams=50)

    # descr_viewer.intersections_ngrams(description_train, symptoms, sympt_vectorizer.sympt_index, 0, sample_size=50)
    # descr_viewer.intersections_ngrams(description_train, symptoms, sympt_vectorizer.sympt_index, 2, sample_size=50)
    # descr_viewer.intersections_ngrams(description_train, symptoms, sympt_vectorizer.sympt_index, 3, sample_size=100)
    # descr_viewer.intersections_ngrams(description_train, symptoms, sympt_vectorizer.sympt_index, 4, sample_size=100)
