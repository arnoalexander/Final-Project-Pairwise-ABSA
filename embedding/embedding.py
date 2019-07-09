"""
Represent word into dense vector
"""

import gensim
import numpy as np


class FastTextEmbedding:

    """
    Wrapper class for gensim FastText
    """

    def __init__(self, *args, size=25, **kwargs):
        self.size = size
        self.model = gensim.models.FastText(*args, size=size, **kwargs)

    def build_vocab(self, *args, **kwargs):
        self.model.build_vocab(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def get_vector_word(self, word):
        try:
            result = self.model.wv[word].copy()
        except KeyError:
            result = np.zeros(self.size, np.float32)
        return result

    def get_vector_sentence(self, sentence, use_norm=False):
        sentence_found = 0
        result = np.zeros(self.size, np.float32)
        for word in sentence:
            try:
                word_vector = self.model.wv[word].copy()
                if use_norm:
                    word_vector_length = np.linalg.norm(word_vector)
                    if word_vector_length != 0.0:
                        word_vector /= word_vector_length
                result += word_vector
                sentence_found += 1
            except KeyError:
                pass
        return result / max(1, sentence_found)


if __name__ == '__main__':
    document = [["the", "quick", "brown", "fox"], ["jumps", "over", "a", "lazy", "dog"]]
    test_word = "test"
    test_sentence = ["there", "is", "a", "test", "duck"]
    embedding = FastTextEmbedding(min_count=1, size=2, min_n=3)
    embedding.build_vocab(document)
    print(embedding.get_vector_word(test_word))
    print(embedding.get_vector_sentence(test_sentence))
