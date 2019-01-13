from tools.utils import Utils


class DescriptionParser:
    def __init__(self, *, config_data):
        self.config_data = config_data
        self.utils = Utils()
        self.max_ngrams_size = self.config_data.get("max_ngrams_size")

    def preprocessed(self, description):
        correction = self.utils.spell_checking(description)
        tokens = self.utils.tokenization(description)
        if correction:
            tokens = [correction.get(token, token) for token in tokens]
        morpho_info = [self.utils.get_pos(tokens[index], index) for index in range(len(tokens))]
        return morpho_info

    def parse(self, description, symptoms_block, ngrams, pattern_type='keywords'):

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
