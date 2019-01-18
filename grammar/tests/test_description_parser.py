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
    """
    Класс для тестирования rule-based подхода. Используем unittest. Тестируем каждое правило - измеряем метрику.
    После тестируем все правила на всех данных.

    Каждый тест на правило состоит из следующих компонентов:
        * parser_cycle. Мы берем только опсиания с целевым симптомом и применяем наше правило. Смотрим на точность - в
        идеале мы должны для сделать правило, которое находит симптомы во всех целевых описаниях. То есть, теоретически
        точность должна быть близка к 1. Полноту смысла мерить нет, так как мы работаем с одним правилом.
        В ходе тестирования мы можем изменить набор ключевых слов на основании анализа логов. Посмотреть, как правило
        работает на тех или иных кейсах.

        * parse_cycle_all_data. Правило прогоняется на всех данных. Смотрим на False Positive случаи - корректируем
        правила. На данном этапе можно мерить все три метрики: точность, полнота и jaccard_measure.
    """

    def setUp(self):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../') + '/data/raw_data/')

        self.experiment_config = Config(file_name=config.CONFIG_PATH)
        self.descr_parser = DescriptionParser(config_data=self.experiment_config)
        data_loader = DataLoader(config_data=self.experiment_config, data_path=data_path)
        self.utils = Utils()

        self.max_ngrams_size = self.experiment_config.get('max_ngrams_size')
        self.data = data_loader.load_chunks()
        self.ep = 0.0001

        self.verbose = False
        self.preprocessed_data = [
            {
                "initial_element": element,
                "preprocessed_element": self.descr_parser.preprocessed(element.description)
            } for element in tqdm(self.data, desc='data_preprocessing')]

    def jaccard_measure(self, y_true, y_pred):
        intersection = len(set(y_true).intersection(y_pred))
        union = (len(y_true) + len(y_pred)) - intersection
        if intersection and union:
            return float(intersection / union)
        else:
            return 0.0

    def precision(self, y_true, y_pred):
        return len(set(y_true).intersection(set(y_pred))) / (len(y_pred) + self.ep)

    def recall(self, y_true, y_pred):
        return len(set(y_true).intersection(set(y_pred))) / (len(y_true) + self.ep)

    def parse_cycle_all_data(self, pattern_name, verbose=True):
        prec = []
        recall = []
        jacc = []

        if verbose:
            self.utils.init_logging(pattern_name)

        for index_element in tqdm(range(len(self.data))):

            fact_symptoms = []
            preprocessed_data = self.preprocessed_data[index_element]["preprocessed_element"]
            true_symptoms = self.preprocessed_data[index_element]["initial_element"].symptoms

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

            prec_obj = self.precision(true_symptoms, fact_symptoms)
            rec_obj = self.recall(true_symptoms, fact_symptoms)
            jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

            if verbose:
                logging.info('element: ' + str(self.preprocessed_data[index_element]["initial_element"]))
                logging.info('preprocessed_data: ' + str(preprocessed_data))
                logging.info('keywords: ' + str(key_word_ngrams))
                logging.info('delimiters: ' + str(delim_word_ngrams))
                logging.info('true_symptoms: ' + str(true_symptoms))
                logging.info('fact_symptoms: ' + str(fact_symptoms))
                logging.info("Pr: {:f} Rec: {:f}; Jac: {:f};".format(prec_obj, rec_obj, jacc_obj))
                logging.info('\n')

            prec.append(prec_obj)
            recall.append(rec_obj)
            jacc.append(jacc_obj)

        prec = sum(prec) / len(prec)
        recall = sum(recall) / len(recall)
        jacc = sum(jacc) / len(jacc)

        if verbose:
            logging.info("Pr: {:f} Rec: {:f}; Jac: {:f};".format(prec, recall, jacc))

        return {"prec": prec, "recall": recall, "jacc": jacc}

    def parser_cycle(self, pattern_name, verbose=True):
        count_descr_with_sympt = 0
        prec = []
        recall = []
        jacc = []

        if verbose:
            self.utils.init_logging(pattern_name)

        for index_element in tqdm(range(len(self.data))):

            if set(self.experiment_config.get(pattern_name)["symptoms"]).intersection(
                    set(self.preprocessed_data[index_element]["initial_element"].symptoms)):

                count_descr_with_sympt += 1
                true_symptoms = self.preprocessed_data[index_element]["initial_element"].symptoms
                fact_symptoms = []

                preprocessed_data = self.preprocessed_data[index_element]["preprocessed_element"]

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

                prec_obj = self.precision(true_symptoms, fact_symptoms)
                rec_obj = self.recall(true_symptoms, fact_symptoms)
                jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

                if verbose:
                    logging.info('element: ' + str(self.preprocessed_data[index_element]["initial_element"]))
                    logging.info('preprocessed_data: ' + str(preprocessed_data))
                    logging.info('keywords: ' + str(key_word_ngrams))
                    logging.info('delimiters: ' + str(delim_word_ngrams))
                    logging.info('fact_symptoms: ' + str(fact_symptoms))
                    logging.info("Pr: {:f} Rec: {:f}; Jac: {:f};".format(prec_obj, rec_obj, jacc_obj))
                    logging.info('\n')

                prec.append(prec_obj)
                recall.append(rec_obj)
                jacc.append(jacc_obj)

        prec = sum(prec) / len(prec)
        recall = sum(recall) / len(recall)
        jacc = sum(jacc) / len(jacc)

        if verbose:
            logging.info("Count_descr: {:d}; Pr: {:f} Rec: {:f}; Jac: {:f};".format(
                count_descr_with_sympt, prec, recall, jacc))

        return {"count_descr_with_sympt": count_descr_with_sympt, "prec": prec, "recall": recall, "jacc": jacc}

    def parse_group_1(self):

        true_params = {
            'true_count_descr_with_sympt': 30,
            'true_precision': 0.966570,
            'true_recall': 0.376094,
            'true_jac': 0.376111
        }

        true_params_all_data = {
            'true_precision': 0.065016,
            'true_recall': 0.025298,
            'true_jac': 0.025299
        }

        fact_results = self.parser_cycle('боль_в_груди', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('боль_в_груди', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_2(self):

        true_params = {
            'true_count_descr_with_sympt': 62,
            'true_precision': 0.983773,
            'true_recall': 0.357531,
            'true_jac': 0.357546
        }

        true_params_all_data = {
            'true_precision': 0.136758,
            'true_recall': 0.049702,
            'true_jac': 0.049704
        }

        fact_results = self.parser_cycle('головная_боль', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('головная_боль', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_3(self):

        true_params = {
            'true_count_descr_with_sympt': 27,
            'true_precision': 0.888800,
            'true_recall': 0.258060,
            'true_jac': 0.258069
        }

        true_params_all_data = {
            'true_precision': 0.053806,
            'true_recall': 0.015622,
            'true_jac': 0.015623
        }

        fact_results = self.parser_cycle('насморк', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('насморк', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_4(self):

        true_params = {
            'true_count_descr_with_sympt': 19,
            'true_precision': 0.894647,
            'true_recall': 0.281569,
            'true_jac': 0.281579
        }

        true_params_all_data = {
            'true_precision': 0.038113,
            'true_recall': 0.011995,
            'true_jac': 0.011996
        }

        fact_results = self.parser_cycle('рвота', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('рвота', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_5(self):

        true_params = {
            'true_count_descr_with_sympt': 22,
            'true_precision': 0.954450,
            'true_recall': 0.327261,
            'true_jac': 0.327273
        }

        true_params_all_data = {
            'true_precision': 0.047080,
            'true_recall': 0.016143,
            'true_jac': 0.016143
        }

        fact_results = self.parser_cycle('тошнота', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('тошнота', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_6(self):

        true_params = {
            'true_count_descr_with_sympt': 22,
            'true_precision': 0.954450,
            'true_recall': 0.336351,
            'true_jac': 0.336364
        }

        true_params_all_data = {
            'true_precision': 0.047080,
            'true_recall': 0.016591,
            'true_jac': 0.016592
        }

        fact_results = self.parser_cycle('пульсирует в висках', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('пульсирует в висках', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_7(self):

        true_params = {
            'true_count_descr_with_sympt': 19,
            'true_precision': 0.947274,
            'true_recall': 0.305690,
            'true_jac': 0.305702
        }

        true_params_all_data = {
            'true_precision': 0.040355,
            'true_recall': 0.013023,
            'true_jac': 0.013023
        }

        fact_results = self.parser_cycle('слабость', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('слабость', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_8(self):

        true_params = {
            'true_count_descr_with_sympt': 39,
            'true_precision': 0.897346,
            'true_recall': 0.253868,
            'true_jac': 0.253877
        }

        true_params_all_data = {
            'true_precision': 0.078467,
            'true_recall': 0.022199,
            'true_jac': 0.022200
        }

        fact_results = self.parser_cycle('повышение температуры', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('повышение температуры', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_9(self):

        true_params = {
            'true_count_descr_with_sympt': 17,
            'true_precision': 0.882265,
            'true_recall': 0.291656,
            'true_jac': 0.291667
        }

        true_params_all_data = {
            'true_precision': 0.033629,
            'true_recall': 0.011117,
            'true_jac': 0.011117
        }

        fact_results = self.parser_cycle('чихание', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('чихание', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_10(self):

        true_params = {
            'true_count_descr_with_sympt': 14,
            'true_precision': 0.928479,
            'true_recall': 0.454732,
            'true_jac': 0.454762
        }

        true_params_all_data = {
            'true_precision': 0.029145,
            'true_recall': 0.014274,
            'true_jac': 0.014275
        }

        fact_results = self.parser_cycle('одышка', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('одышка', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_11(self):

        true_params = {
            'true_count_descr_with_sympt': 14,
            'true_precision': 0.785636,
            'true_recall': 0.366051,
            'true_jac': 0.366071
        }

        true_params_all_data = {
            'true_precision': 0.024661,
            'true_recall': 0.011490,
            'true_jac': 0.011491
        }

        fact_results = self.parser_cycle('озноб', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('озноб', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_12(self):

        true_params = {
            'true_count_descr_with_sympt': 24,
            'true_precision': 0.874913,
            'true_recall': 0.225787,
            'true_jac': 0.225794
        }

        true_params_all_data = {
            'true_precision': 0.047080,
            'true_recall': 0.012150,
            'true_jac': 0.012150
        }

        fact_results = self.parser_cycle('температура 38 градусов и больше', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('температура 38 градусов и больше', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_13(self):

        true_params = {
            'true_count_descr_with_sympt': 16,
            'true_precision': 0.999900,
            'true_recall': 0.347901,
            'true_jac': 0.347917
        }

        true_params_all_data = {
            'true_precision': 0.035871,
            'true_recall': 0.012481,
            'true_jac': 0.012481
        }

        fact_results = self.parser_cycle('плохой сон', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('плохой сон', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_14(self):

        true_params = {
            'true_count_descr_with_sympt': 20,
            'true_precision': 0.849915,
            'true_recall': 0.334984,
            'true_jac': 0.335000
        }

        true_params_all_data = {
            'true_precision': 0.038113,
            'true_recall': 0.015022,
            'true_jac': 0.015022
        }

        fact_results = self.parser_cycle('боль внезапная', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('боль внезапная', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def parse_group_15(self):

        true_params = {
            'true_count_descr_with_sympt': 23,
            'true_precision': 0.739057,
            'true_recall': 0.271366,
            'true_jac': 0.271377
        }

        true_params_all_data = {
            'true_precision': 0.038113,
            'true_recall': 0.013994,
            'true_jac': 0.013995
        }

        fact_results = self.parser_cycle('сильный зуд', verbose=self.verbose)

        self.assertEqual(true_params['true_count_descr_with_sympt'], fact_results['count_descr_with_sympt'])
        self.assertEqual(true_params['true_precision'], round(fact_results['prec'], 6))
        self.assertEqual(true_params['true_recall'], round(fact_results['recall'], 6))
        self.assertEqual(true_params['true_jac'], round(fact_results['jacc'], 6))

        fact_results_all_data = self.parse_cycle_all_data('сильный зуд', verbose=self.verbose)

        self.assertEqual(true_params_all_data['true_precision'], round(fact_results_all_data['prec'], 6))
        self.assertEqual(true_params_all_data['true_recall'], round(fact_results_all_data['recall'], 6))
        self.assertEqual(true_params_all_data['true_jac'], round(fact_results_all_data['jacc'], 6))

    def test_grammar(self):

        true_params_all_data = {
            'true_precision': 0.419476,
            'true_recall': 0.261101,
            'true_jac': 0.234636
        }

        prec = []
        recall = []
        jacc = []

        self.utils.init_logging('grammar_test')

        patterns = ['боль_в_груди', 'головная_боль', 'насморк', 'рвота', 'тошнота', 'пульсирует в висках',
                    'слабость',
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

            prec_obj = self.precision(true_symptoms, fact_symptoms)
            rec_obj = self.recall(true_symptoms, fact_symptoms)
            jacc_obj = self.jaccard_measure(true_symptoms, fact_symptoms)

            if self.verbose:
                logging.info('element: ' + str(element))
                logging.info('preprocessed_data: ' + str(preprocessed_data))
                logging.info('true_symptoms: ' + str(true_symptoms))
                logging.info('fact_symptoms: ' + str(fact_symptoms))
                logging.info("Pr: {:f} Rec: {:f}; Jac: {:f};".format(prec_obj, rec_obj, jacc_obj))
                logging.info('\n')

            prec.append(prec_obj)
            recall.append(rec_obj)
            jacc.append(jacc_obj)

        prec = sum(prec) / len(prec)
        recall = sum(recall) / len(recall)
        jacc = sum(jacc) / len(jacc)

        logging.info("Pr: {:f} Rec: {:f}; Jac: {:f};".format(prec, recall, jacc))

        self.assertEqual(true_params_all_data['true_precision'], round(prec, 6))
        self.assertEqual(true_params_all_data['true_recall'], round(recall, 6))
        self.assertEqual(true_params_all_data['true_jac'], round(jacc, 6))


if __name__ == '__main__':
    unittest.main()
