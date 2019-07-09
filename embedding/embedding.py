"""
Represent word into dense vector
"""

import gensim
import numpy as np
import definition


class FastTextEmbedding:

    """
    Wrapper class for gensim FastText
    """

    def __init__(self, *args, **kwargs):
        self.model = gensim.models.FastText(*args, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = gensim.models.FastText.load(path)

    def build_vocab(self, *args, **kwargs):
        self.model.build_vocab(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def get_vector_word(self, word):
        try:
            result = self.model.wv[word].copy()
        except KeyError:
            result = np.zeros(self.model.wv.vector_size, np.float32)
        return result

    def get_vector_sentence(self, sentence, use_norm=False):
        sentence_found = 0
        result = np.zeros(self.model.wv.vector_size, np.float32)
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
    test_word = "thick"
    test_sentence = ["there", "is", "a", "test", "duck"]
    embedding = FastTextEmbedding(min_count=1, size=5, min_n=3)
    embedding.build_vocab(document)
    embedding.save(definition.MODEL_EMBEDDING_SAMPLEFASTTEXT)
    new_embedding = FastTextEmbedding()
    new_embedding.load(definition.MODEL_EMBEDDING_SAMPLEFASTTEXT)
    print(new_embedding.get_vector_word(test_word))
    print(new_embedding.get_vector_sentence(test_sentence))
