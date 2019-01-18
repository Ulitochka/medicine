from tools.utils import Utils


class DescriptionParser:
    """
    Класс реализующий rule-based подход для выявления симпотомов в описаниях состояний, составленных самими пациентами.
    """

    def __init__(self, *, config_data):
        self.config_data = config_data
        self.utils = Utils()
        self.max_ngrams_size = self.config_data.get("max_ngrams_size")

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
        morpho_info = [self.utils.get_pos(tokens[index], index) for index in range(len(tokens))]
        return morpho_info

    def parse(self, description, symptoms_block, ngrams, pattern_type='keywords'):
        """
        Метод принимающий на вход сгенерированные из описания нграммы, список ключевых слов для определения симптома,
        разделенных на несколько категорий: object - объект, feel - ощущения, place - место на теле человека, operators
        - операторы, в нашем случае это будут различные служебные части речи (предлоги, союзы).

        Суть метода в нахождении пересечений различных слов из нграммы со словами из категорий симптома.
        В конце у нас получается бинарный вектор для каждой нграммы. Если все компоненты этого вектора (то есть пересе-
        чения с определенными ключевыми леммами из категорий) совпадают с переменной kw_status (так как у какого то
        симптома может не быть ключевых слов для какой-либо категории), то мы можем отнести симптом к описанию.
        Таким образом, каждый набор symptoms_block представляет собой правило для нахождения определенного типа симптома.

        Различные значения аргумента pattern_type.
        Существует два вида паттернов: keywords - ключевые слова разных категорий; delimiters - так как в разметке
        описаний много противоречивых моментов, то они помогают решать некоторые противоречивые случаи, когда по одним
        и тем же словам можно отнести разные симпотомы. По сути это keywords, они используются следующим образом: так
        как правила применяются независимо, хотя интересно было бы сделать иерархию, то симптом присваивается только
        в следующем случае: если есть совпадение с keywords и нет совпадений с delimiters, то симптом присваивается;

        :param description:
        :param symptoms_block:
        :param ngrams:
        :param pattern_type:
        :return:
        """

        key_word_ngrams = []

        kw_status = {
            "object": True if symptoms_block[pattern_type]['object'] else False,
            "feel": True if symptoms_block[pattern_type]['feel'] else False,
            "place": True if symptoms_block[pattern_type]['place'] else False,
            "operators": True if symptoms_block[pattern_type]['operators'] else False
        }

        if [s for s in kw_status if kw_status[s]]:

            if not ngrams:
                ngrams = [self.utils.ngrams(description, n) for n in range(2, self.max_ngrams_size)]

            n_grams = [
                {"ngrams": [t['normal_form'] for t in ngrams],
                 "object": False,
                 "feel": False,
                 "place": False,
                 "operators": False
                 } for ngram_variant in ngrams for ngrams in ngram_variant]

            for collocation in n_grams:
                for kw in symptoms_block[pattern_type]:
                    for word in symptoms_block[pattern_type][kw]:
                        if word in collocation['ngrams']:
                            collocation[kw] = True

                checking_status = [collocation[kw] == kw_status[kw] for kw in kw_status]
                if all(checking_status):
                    key_word_ngrams.append(collocation['ngrams'])

        return key_word_ngrams
