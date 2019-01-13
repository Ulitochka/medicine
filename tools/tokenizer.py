from nltk import WordPunctTokenizer


class Tokenizer:
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())
