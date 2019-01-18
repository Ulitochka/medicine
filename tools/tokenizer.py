from nltk import WordPunctTokenizer


class Tokenizer:
    """
    Класс осуществляющий токенизацию.
    """

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())
