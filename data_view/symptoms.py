import os
import config
import logging
from itertools import chain
from collections import Counter
from pprint import pprint

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from tools.utils import Utils
from tools.data_loader import DataObject, DataLoader
from tools.symptoms_vectorizer import SymptomsVectorizer
from config.Config_worker import Config


class SymptomsViewer:
    def __init__(self, plot_path):
        self.utils = Utils()
        self.plot_path = plot_path

        self.utils.init_logging('data_view_symptoms')

    def freq_sympt(self, data):
        freq_sympt = Counter(data)
        sympt_freq_distr = pd.DataFrame()
        sympt_freq_distr['symp'] = [el for el in freq_sympt]
        sympt_freq_distr['freq'] = [freq_sympt[el] for el in freq_sympt]
        sympt_freq_distr = sympt_freq_distr.sort_values(by='freq')

        average = round(sympt_freq_distr["freq"].mean(), 0)

        sympt_freq_distr.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e', histtype='barstacked')
        plt.xlabel('Frequency')
        plt.ylabel('Sympt_count')
        plt.grid(axis='y', alpha=1.)
        plt.text(25, 10, r'$\mu=%s, sum=%s$' % (average, sum([freq_sympt[el] for el in freq_sympt])))
        plt.savefig(os.path.join(self.plot_path, 'group_freq_sympt.jpeg'))

        plt.rcParams['figure.figsize'] = 24, 16
        sympt_freq_distr.plot(x='symp', y='freq', kind='bar')
        plt.savefig(os.path.join(self.plot_path, 'freq_sympt.jpeg'))

        logging.info(tabulate(sympt_freq_distr, tablefmt="fancy_grid"))

    def average_sympt_per_descr(self, data):
        sympts = [len(s) for s in data]
        logging.info(tabulate([
            ['max', max(sympts)], ['min', min(sympts)], ['aver', sum(sympts) / len(sympts)]],
            headers=['#sympt_per_descr', 'value'],
            tablefmt="fancy_grid"))

    def sorted_symptoms(self, data):
        pprint(sorted(set([s for el in data for s in el])))

    def groups_sympt_per_descr(self, tables, symptoms):
        tables = {sub_el[-1]: el for el in tables for sub_el in tables[el]}
        gr_sympts = ['&'.join(sorted(set([tables.get(el) for el in s]))) for s in symptoms]
        gr_sympts_counter = Counter(gr_sympts)
        sympt_gr_freq_distr = pd.DataFrame()
        sympt_gr_freq_distr['symp_gr'] = [el for el in gr_sympts_counter]
        sympt_gr_freq_distr['freq'] = [gr_sympts_counter[el] for el in gr_sympts_counter]
        sympt_gr_freq_distr = sympt_gr_freq_distr.sort_values(by='freq')

        logging.info(tabulate(sympt_gr_freq_distr, tablefmt="fancy_grid"))

        return [(index, gr_sympts[index]) for index in range(len(gr_sympts)) if '&' in gr_sympts[index]]


if __name__ == '__main__':
    experiment_config = Config(file_name=config.CONFIG_PATH)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/')
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/plots/')
    tables_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/Tables/')

    data_loader = DataLoader(config_data=experiment_config, data_path=data_path)
    sympt_vectorizer = SymptomsVectorizer(config_data=experiment_config, data_path=tables_path)

    symptoms_viewer = SymptomsViewer(plot_path=plot_path)
    data = data_loader.load_chunks()

    description_train = [el.description for el in data]
    symptoms = [sorted(el.symptoms) for el in data]

    symptoms_viewer.average_sympt_per_descr(symptoms)
    symptoms_viewer.freq_sympt(chain(*symptoms))
    symptoms_viewer.sorted_symptoms(symptoms)

    mix_sympt = symptoms_viewer.groups_sympt_per_descr(sympt_vectorizer.tables, symptoms)

