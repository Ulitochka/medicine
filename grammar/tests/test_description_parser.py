import unittest
import config
import logging
import os
from tqdm import tqdm

from tools.data_loader import DataLoader
from config.Config_worker import Config
from grammar.description_parser import DescriptionParser
from tools.utils import Utils


class TestSymptomParser(unittest.TestCase):
    def setUp(self):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../') + '/data/raw_data/')

        self.experiment_config = Config(file_name=config.CONFIG_PATH)
        self.descr_parser = DescriptionParser(config_data=self.experiment_config)
        data_loader = DataLoader(config_data=self.experiment_config, data_path=data_path)
        self.utils = Utils()

        self.max_ngrams_size = self.experiment_config.get('max_ngrams_size')
        self.data = data_loader.load_chunks()

    def jaccard_measure(self, y_true, y_pred):
        intersection = len(set(y_true).intersection(y_pred))
        union = (len(y_true) + len(y_pred)) - intersection
        if intersection and union:
            return float(intersection / union)
        else:
            return 0.0

    def parse_cycle_all_data(self, pattern_name, verbose=False):
        jacc = []

        if verbose:
            self.utils.init_logging(pattern_name)

        for element in tqdm(self.data):
            true_symptoms = element.symptoms
            fact_symptoms = []

            preprocessed_data = self.descr_parser.preprocessed(element.description)

            key_word_ngrams = self.descr_parser.parse(
                preprocessed_data,
                self.experiment_config.get(pattern_name),
                ngrams=None,
                pattern_type='keywords')

            delim_word_ngrams = self.descr_parser.parse(
                preprocessed_data,
                self.experiment_config.get(pattern_name),
                ngrams=None,
                pattern_type='delimiters')

            if key_word_ngrams and not delim_word_ngrams:
                fact_symptoms = self.experiment_config.get(pattern_name)["symptoms"]

            jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

            if verbose:
                logging.info('element: ' + str(element))
                logging.info('preprocessed_data: ' + str(preprocessed_data))
                logging.info('keywords: ' + str(key_word_ngrams))
                logging.info('delimiters: ' + str(delim_word_ngrams))
                logging.info('true_symptoms: ' + str(true_symptoms))
                logging.info('fact_symptoms: ' + str(fact_symptoms))
                logging.info("Jac: {:f};".format(jacc_obj))
                logging.info('\n')

            jacc.append(jacc_obj)

        jacc = sum(jacc) / len(jacc)

        if verbose:
            logging.info("Jac: {:f};".format(jacc))

        return {"jacc": jacc}

    def parser_cycle(self, pattern_name, verbose=False):
        count_descr_with_sympt = 0
        jacc = []

        if verbose:
            self.utils.init_logging(pattern_name)

        for element in tqdm(self.data):

            if set(self.experiment_config.get(pattern_name)["symptoms"]).intersection(set(element.symptoms)):

                count_descr_with_sympt += 1
                true_symptoms = element.symptoms
                fact_symptoms = []

                preprocessed_data = self.descr_parser.preprocessed(element.description)

                key_word_ngrams = self.descr_parser.parse(
                    preprocessed_data,
                    self.experiment_config.get(pattern_name),
                    ngrams=None,
                    pattern_type='keywords')

                delim_word_ngrams = self.descr_parser.parse(
                    preprocessed_data,
                    self.experiment_config.get(pattern_name),
                    ngrams=None,
                    pattern_type='delimiters')

                if key_word_ngrams and not delim_word_ngrams:
                    fact_symptoms = self.experiment_config.get(pattern_name)["symptoms"]

                jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

                if verbose:
                    logging.info('element: ' + str(element))
                    logging.info('preprocessed_data: ' + str(preprocessed_data))
                    logging.info('keywords: ' + str(key_word_ngrams))
                    logging.info('delimiters: ' + str(delim_word_ngrams))
                    logging.info('fact_symptoms: ' + str(fact_symptoms))
                    logging.info("Jac: {:f};".format(jacc_obj))
                    logging.info('\n')

                jacc.append(jacc_obj)

        jacc = sum(jacc) / len(jacc)

        if verbose:
            logging.info("Count_descr: {:d}; Jac: {:f};".format(count_descr_with_sympt, jacc))

        return {"count_descr_with_sympt": count_descr_with_sympt, "jacc": jacc}

    def test_parse_group_1(self):

        true_params = {
            'true_count_descr_with_sympt': 30,
            'true_jac': 0.376111
        }

        true_params_all_data = {
            'true_jac': 0.025299
        }

        fact_results = self.parser_cycle('боль_в_груди', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('боль_в_груди', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_2(self):

        true_params = {
            'true_count_descr_with_sympt': 62,
            'true_jac':  0.357546
        }

        true_params_all_data = {
            'true_jac': 0.049704
        }

        fact_results = self.parser_cycle('головная_боль', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('головная_боль', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_3(self):

        true_params = {
            'true_count_descr_with_sympt': 27,
            'true_jac':  0.258069
        }

        true_params_all_data = {
            'true_jac': 0.015623
        }

        fact_results = self.parser_cycle('насморк', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('насморк', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_4(self):

        true_params = {
            'true_count_descr_with_sympt': 19,
            'true_jac':  0.281579
        }

        true_params_all_data = {
            'true_jac': 0.011996
        }

        fact_results = self.parser_cycle('рвота', verbose=False)
        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('рвота', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_5(self):

        true_params = {
            'true_count_descr_with_sympt': 22,
            'true_jac':  0.327273
        }

        true_params_all_data = {
            'true_jac': 0.016143
        }

        fact_results = self.parser_cycle('тошнота', verbose=False)
        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('тошнота', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_6(self):

        true_params = {
            'true_count_descr_with_sympt': 22,
            'true_jac': 0.336364
        }

        true_params_all_data = {
            'true_jac': 0.016592
        }

        fact_results = self.parser_cycle('пульсирует в висках', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('пульсирует в висках', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_7(self):

        true_params = {
            'true_count_descr_with_sympt': 19,
            'true_jac': 0.305702
        }

        true_params_all_data = {
            'true_jac': 0.013023
        }

        fact_results = self.parser_cycle('слабость', verbose=False)
        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('слабость', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_8(self):

        true_params = {
            'true_count_descr_with_sympt': 39,
            'true_jac': 0.253877
        }

        true_params_all_data = {
            'true_jac': 0.022200
        }

        fact_results = self.parser_cycle('повышение температуры', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('повышение температуры', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_9(self):

        true_params = {
            'true_count_descr_with_sympt': 17,
            'true_jac': 0.291667
        }

        true_params_all_data = {
            'true_jac': 0.011117
        }

        fact_results = self.parser_cycle('чихание', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('чихание', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_10(self):

        true_params = {
            'true_count_descr_with_sympt': 14,
            'true_jac': 0.454762
        }

        true_params_all_data = {
            'true_jac': 0.014275
        }

        fact_results = self.parser_cycle('одышка', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('одышка', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_11(self):

        true_params = {
            'true_count_descr_with_sympt': 14,
            'true_jac': 0.366071
        }

        true_params_all_data = {
            'true_jac': 0.011491
        }

        fact_results = self.parser_cycle('озноб', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('озноб', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_12(self):

        true_params = {
            'true_count_descr_with_sympt': 24,
            'true_jac': 0.225794
        }

        true_params_all_data = {
            'true_jac': 0.012150
        }

        fact_results = self.parser_cycle('температура 38 градусов и больше', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('температура 38 градусов и больше', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_13(self):

        true_params = {
            'true_count_descr_with_sympt': 16,
            'true_jac': 0.347917
        }

        true_params_all_data = {
            'true_jac': 0.012481
        }

        fact_results = self.parser_cycle('плохой сон', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('плохой сон', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_14(self):

        true_params = {
            'true_count_descr_with_sympt': 20,
            'true_jac': 0.335000
        }

        true_params_all_data = {
            'true_jac': 0.015022
        }

        fact_results = self.parser_cycle('боль внезапная', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('боль внезапная', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_parse_group_15(self):

        true_params = {
            'true_count_descr_with_sympt': 23,
            'true_jac': 0.271377
        }

        true_params_all_data = {
            'true_jac': 0.013995
        }

        fact_results = self.parser_cycle('сильный зуд', verbose=False)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('сильный зуд', verbose=False)
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_grammar(self):

        verbose = True

        true_params_all_data = {
            'true_jac': 0.234636
        }

        jacc = []

        patterns = ['боль_в_груди', 'головная_боль', 'насморк', 'рвота', 'тошнота', 'пульсирует в висках', 'слабость',
                    'повышение температуры', 'чихание', 'одышка', 'озноб', 'температура 38 градусов и больше',
                    'плохой сон', 'боль внезапная', 'сильный зуд']

        for element in tqdm(self.data):
            true_symptoms = element.symptoms
            fact_symptoms = []

            preprocessed_data = self.descr_parser.preprocessed(element.description)
            n_grams = [self.utils.ngrams(preprocessed_data, n) for n in range(2, self.max_ngrams_size)]

            for p in patterns:

                key_word_ngrams = self.descr_parser.parse(
                    preprocessed_data,
                    self.experiment_config.get(p),
                    ngrams=n_grams,
                    pattern_type='keywords')

                delim_word_ngrams = self.descr_parser.parse(
                    preprocessed_data,
                    self.experiment_config.get(p),
                    ngrams=n_grams,
                    pattern_type='delimiters')

                if key_word_ngrams and not delim_word_ngrams:
                    for s in self.experiment_config.get(p)["symptoms"]:
                        fact_symptoms.append(s)

            jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

            if verbose:
                logging.info('element: ' + str(element))
                logging.info('preprocessed_data: ' + str(preprocessed_data))
                logging.info('true_symptoms: ' + str(true_symptoms))
                logging.info('fact_symptoms: ' + str(fact_symptoms))
                logging.info("Jac: {:f};".format(jacc_obj))
                logging.info('\n')

            jacc.append(jacc_obj)
        jacc = sum(jacc) / len(jacc)

        logging.info("Jac: {:f};".format(jacc))
        self.assertEqual(true_params_all_data['true_jac'], round(jacc, 6))


if __name__ == '__main__':
    unittest.main()
