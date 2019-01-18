import uuid
import time
from collections import OrderedDict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
from nltk.corpus import stopwords

from tools.utils import Utils


class Vectorizers:
    """
    Класс для извлечения признаков для классификации описанйи состояний пациентов.
    """

    def __init__(self, *, config_data):
        self.config_data = config_data
        self.seed = 1024
        self.test_size = config_data.get('test_size')
        self.folds_number = config_data.get('folds_number')
        self.data_id = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8]

        stopWords = set(stopwords.words('russian'))

        self.cv_word = CountVectorizer(ngram_range=(1, 5), analyzer='word', stop_words=stopWords)
        self.tf_idf_word = TfidfVectorizer(ngram_range=(1, 5), analyzer='word')

        self.utils = Utils()

    def fit_vectorizer(self, data):
        self.cv_word.fit(data)
        self.tf_idf_word.fit(data)

    def feature_extract(self, tokens):
        """
        Векторизация текстов описаний.
        :param tokens:
        :return:
        """

        features = OrderedDict(
            [
                ('cv_word', self.cv_word.transform(tokens)),
                ('tf_idf_word', self.tf_idf_word.transform(tokens)),
            ]
        )

        X = None
        for feature in features:
            if X is None:
                X = sparse.csr_matrix.copy(features[feature])
            else:
                X = sparse.hstack((X, features[feature]), format='csc')
        return X
