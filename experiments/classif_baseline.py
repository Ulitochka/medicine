import os
import logging
import config
import warnings

from tabulate import tabulate
import scipy.stats
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_similarity_score
from sklearn.ensemble import GradientBoostingClassifier

from tools.utils import Utils
from config.Config_worker import Config
from feature_extractors.vectorizers_features import Vectorizers


warnings.filterwarnings("ignore")
CV = False
TOP_FEATURES = False
ENS = True


class SymptClassifier:
    def __init__(self, *, config_data, model_params):
        self.config_data = config_data
        self.seed = 1024
        self.candidates = self.config_data.get('candidates_base_line')

        self.encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()

        log_reg = LogisticRegression(
            solver='liblinear',
            C=model_params["C"],
            penalty="l1",
            max_iter=1000,
            verbose=0,
            multi_class="ovr",
            random_state=self.seed
        )

        xgb = GradientBoostingClassifier(
            learning_rate=model_params["learning_rate"],
            n_estimators=int(model_params["n_estimators"]),
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=1.0,
            verbose=1,
            random_state=self.seed

        )

        models = {
            "log_reg": log_reg,
            "xgb": xgb
        }

        if TOP_FEATURES:
            self.model = OneVsRestClassifier(
                Pipeline([('chi2', SelectKBest(chi2, k=50000)),
                          ('log_reg', log_reg)]))
        else:
            self.model = OneVsRestClassifier(
                estimator=models[model_params["model"]],
                n_jobs=-1
            )

    def optimise_xgb(self, x_train, y_train, x_test, y_test, n_estimators, learning_rate):

        def target(n_estimators, learning_rate):
            clf = OneVsRestClassifier(GradientBoostingClassifier(
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                verbose=0,
                random_state=self.seed
        ))
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_test)
            result = f1_score(y_test, predictions, average='weighted')
            return result

        bo = BayesianOptimization(f=target,
                                  pbounds={'n_estimators': n_estimators, 'learning_rate': learning_rate},
                                  random_state=self.seed)

        bo.maximize(init_points=5, n_iter=20)
        max_target = max([t['target'] for t in bo.res])
        aim_params = [el for el in bo.res if el['target'] == max_target][0]
        return aim_params

    def optimise_log_reg(self, x_train, y_train, x_test, y_test, C):

        def target(C):
            clf = OneVsRestClassifier(LogisticRegression(
                penalty='l1', max_iter=1000, C=C, multi_class='ovr', random_state=self.seed))
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_test)
            result = f1_score(y_test, predictions, average='weighted')
            return result

        bo = BayesianOptimization(f=target, pbounds={'C': C}, random_state=self.seed)
        bo.maximize(init_points=5, n_iter=50)
        max_target = max([t['target'] for t in bo.res])
        aim_params = [el for el in bo.res if el['target'] == max_target][0]
        return aim_params


if __name__ == '__main__':

    data_id = '2019_01_12-08_20_42-0f24f622_k_folds.pkl'

    experiment_config = Config(file_name=config.CONFIG_PATH)
    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/split_path/')
    tables_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/Tables/')

    utils = Utils()

    folds = utils.load_bin_data(os.path.join(split_path, data_id))
    folds_params = experiment_config.get('folds_params')

    utils.init_logging(file_name='SymptClassifier')

    folds_score = []

    for f in folds:

        logging.info('*' * 100)

        vectorizers = Vectorizers(config_data=experiment_config)

        model_params = folds_params[str(f)]
        model_params["model"] = "log_reg"
        sympt_classifier_1 = SymptClassifier(config_data=experiment_config, model_params=folds_params[str(f)])

        description_train = folds[f]['description_train']
        description_test = folds[f]['description_test']
        sympt_train = folds[f]['sympt_train']
        sympt_test = folds[f]['sympt_test']

        vectorizers.fit_vectorizer(description_train + description_test)
        x_train = vectorizers.feature_extract(description_train)
        x_test = vectorizers.feature_extract(description_test)

        logging.info(vectorizers.cv_word.get_params())
        logging.info(vectorizers.tf_idf_word.get_params())

        s_train = [s for el in sympt_train for s in el]
        s_test = [s for el in sympt_test for s in el]

        sympt_classifier_1.encoder.fit(s_train + s_test)
        encoded_train_Y = [sympt_classifier_1.encoder.transform(el) for el in sympt_train]
        encoded_test_Y = [sympt_classifier_1.encoder.transform(el) for el in sympt_test]

        sympt_classifier_1.mlb.fit(encoded_train_Y + encoded_test_Y)
        y_train = sympt_classifier_1.mlb.transform(encoded_train_Y)
        y_test = sympt_classifier_1.mlb.transform(encoded_test_Y)

        logging.info("X_train features: {:s};".format(str(x_train.shape)))
        logging.info('sympt_train_labels: ' + str(y_train.shape))
        logging.info("X_test features: {:s};".format(str(x_test.shape)))
        logging.info('sympt_test_labels: ' + str(y_test.shape))
        logging.info(sympt_classifier_1.model.get_params())

        if CV:
            best_params = sympt_classifier_1.optimise_xgb(
                x_train, y_train, x_test, y_test, n_estimators=(30.0, 100.0), learning_rate=(0.05, 1.0))

            logging.info('best_params: ' + str(best_params))

        elif ENS:
            sympt_classifier_1.model.fit(x_train, y_train)

            model_params["model"] = "xgb"
            sympt_classifier_2 = SymptClassifier(config_data=experiment_config, model_params=folds_params[str(f)])
            sympt_classifier_2.model.fit(x_train, y_train)

            predictions_1 = sympt_classifier_1.model.predict_proba(x_test)
            predictions_2 = sympt_classifier_2.model.predict_proba(x_test)
            out = np.array([predictions_1, predictions_2])
            predictions = scipy.stats.gmean(out)
            predictions = (predictions > 0.5).astype(np.int)
            folds_score.append((f,
                                f1_score(y_test, predictions, average='weighted'),
                                recall_score(y_test, predictions, average='weighted'),
                                precision_score(y_test, predictions, average='weighted'),
                                jaccard_similarity_score(y_test, predictions)))

        else:
            sympt_classifier_1.model.fit(x_train, y_train)
            predictions = sympt_classifier_1.model.predict(x_test)

            folds_score.append((f,
                                f1_score(y_test, predictions, average='weighted'),
                                recall_score(y_test, predictions, average='weighted'),
                                precision_score(y_test, predictions, average='weighted'),
                                jaccard_similarity_score(y_test, predictions)))

    if not CV:

        result = pd.DataFrame()
        result['fold_#'] = [el[0] for el in folds_score]
        result['precision'] = [el[3] for el in folds_score]
        result['recall'] = [el[2] for el in folds_score]
        result['f1'] = [el[1] for el in folds_score]
        result['jaccard'] = [el[-1] for el in folds_score]

        logging.info(tabulate(result, headers='keys', tablefmt='psql'))

        logging.info("CV score f1 : Mean - %.7f | Std - %.7f | Min - %.7f | Max - %.7f" % (
            result['f1'].mean(), result['f1'].std(), result['f1'].min(), result['f1'].max()))

        logging.info("CV score jaccard : Mean - %.7f | Std - %.7f | Min - %.7f | Max - %.7f" % (
            result['jaccard'].mean(), result['jaccard'].std(), result['jaccard'].min(), result['jaccard'].max()))
