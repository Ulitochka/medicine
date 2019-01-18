import os
import config
import time
import uuid

from sklearn.model_selection import KFold

from tools.utils import Utils
from config.Config_worker import Config
from tools.data_loader import DataLoader


class DataSplitter:
    """
    Класс реализующий препроцессинг данных и их разделение на обучающие и тестовые. Разделение осуществляется по 10
    фолдам, так как данных мало и классов разное количество.
    """

    def __init__(self, *, config_data, split_path):
        self.config_data = config_data
        self.seed = 1024
        self.test_size = config_data.get('test_size')
        self.folds_number = config_data.get('folds_number')
        self.data_id = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8]

        self.utils = Utils()
        self.split_path = split_path

    def preprocessed(self, description):
        """
        Метод осуществляющий препроцессинг данных:
            исправление ошибок
            токенизация
            лемматизация
        :param description:
        :return:
        """

        correction = self.utils.spell_checking(description)
        tokens = self.utils.tokenization(description)
        if correction:
         tokens = [correction.get(token, token) for token in tokens]
        morpho_info = [self.utils.get_pos(tokens[index], index)['normal_form'] for index in range(len(tokens))]
        return morpho_info

    def get_split(self, data):
        """
        Метод разделяющий данные на обучающие множества.
        :param data:
        :return:
        """

        folds = dict()

        fold_counter = 0
        skf = KFold(n_splits=self.folds_number, shuffle=True, random_state=self.seed).split(data)
        for ind_tr, ind_te in skf:
            x_train = [data[tr_ind] for tr_ind in ind_tr]
            x_test = [data[tr_ind] for tr_ind in ind_te]
            print("X_train size: {:d}; X_test_size: {:d};".format(len(x_train), len(x_test)))

            description_train = [el.description for el in x_train]
            sympt_train = [el.symptoms for el in x_train]
            description_test = [el.description for el in x_test]
            sympt_test = [el.symptoms for el in x_test]

            description_train = [' '.join(self.preprocessed(el)) for el in description_train]
            description_test = [' '.join(self.preprocessed(el)) for el in description_test]

            fold_counter += 1
            folds[fold_counter] = {
                'description_train': description_train,
                'description_test': description_test,
                'sympt_train': sympt_train,
                'sympt_test': sympt_test
            }

        self.utils.save_binary(folds, os.path.join(self.split_path, self.data_id + '_k_folds.pkl'))


if __name__ == '__main__':
    experiment_config = Config(file_name=config.CONFIG_PATH)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/')
    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/split_path/')

    data_loader = DataLoader(config_data=experiment_config, data_path=data_path)
    data_splitter = DataSplitter(config_data=experiment_config, split_path=split_path)

    data = data_loader.load_chunks()
    data_splitter.get_split(data)
