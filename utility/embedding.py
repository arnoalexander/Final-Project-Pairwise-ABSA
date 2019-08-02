"""
Represent word into dense vector
"""

from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Embedding:

    """
    Wrapper class for gensim FastText
    """

    # Constants
    DEFAULT_COS_DISTANCE = 0

    def __init__(self, *args, model_base=None, model_filename=None, **kwargs):
        """
        Initialize object.
        Set model_filename (path to saved base embedding model) OR model_base(initialized embedding model).
        If both are specified, model_base is preferred. If none, new model will be generated based on args and kwargs.
        """
        if model_base is not None:
            self.model = model_base
        elif model_filename is not None:
            self.load(path=model_filename)
        else:
            self.model = FastText(*args, **kwargs)

    def save(self, path):
        """
        Save model to path
        """
        self.model.save(path)

    def load(self, path):
        """
        Load model from path
        """
        self.model = FastText.load(path)

    def build_vocab(self, *args, **kwargs):
        """
        Gensim build_vocab wrapper
        """
        self.model.build_vocab(*args, **kwargs)

    def train(self, *args, total_examples=None, epochs=None, verbose=False, **kwargs):
        """
        Gensim train wrapper
        """
        if total_examples is None:
            total_examples = self.model.corpus_count
        if epochs is None:
            epochs = self.model.epochs
        if verbose:
            kwargs['callbacks'] = [EmbeddingEpochCallback()]
        self.model.train(*args, total_examples=total_examples, epochs=epochs, **kwargs)

    def get_vector_word(self, word, use_norm=False, handle_oov=True):
        """
        Get vector representation of a word
        """
        if handle_oov:
            try:
                result = self.model.wv.word_vec(word=word, use_norm=use_norm)
            except KeyError:
                result = np.zeros(self.model.wv.vector_size, np.float32)
            return result
        return self.model.wv.word_vec(word=word, use_norm=use_norm)

    def get_vector_sentence(self, sentence, min_n=3, max_n=3, use_norm_ngram_word=False, use_norm_word=False,
                            use_norm_ngram_char=False, handle_oov=True):
        """
        Get vector representation of a sentence
        """
        if min_n > max_n or min_n > len(sentence):
            min_n = len(sentence)
            max_n = len(sentence)
        elif max_n > len(sentence):
            max_n = len(sentence)
        ngrams = []
        for n in range(min_n, max_n + 1):
            ngrams += self._get_ngram_word(sentence=sentence, n=n)
        ngrams_found = 0
        result = np.zeros(self.model.wv.vector_size, np.float32)
        for ngram in ngrams:
            try:
                ngram_vector = self._get_vector_ngram_words(
                    ngram_words=ngram, use_norm_word=use_norm_word, use_norm_ngram_char=use_norm_ngram_char,
                    handle_oov=False)
                if use_norm_ngram_word:
                    ngram_vector = self.normalize_vector(ngram_vector)
                result += ngram_vector
                ngrams_found += 1
            except KeyError:
                pass
        if not handle_oov and ngrams_found == 0:
            raise KeyError('all word level n-grams are absent from model')
        else:
            return result / max(1, ngrams_found)

    @staticmethod
    def normalize_vector(vec):
        vec_length = np.linalg.norm(vec)
        if vec_length == 0.0:
            return vec
        return vec / vec_length

    @staticmethod
    def cosine_distance(vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    @staticmethod
    def _get_ngram_word(sentence, n):
        if n > len(sentence):
            n = len(sentence)
        elif n <= 0:
            return []
        result = []
        for idx_start in range(0, len(sentence) - n + 1):
            result.append(sentence[idx_start:idx_start + n])
        return result

    def _get_vector_ngram_words(self, ngram_words, use_norm_word=False, use_norm_ngram_char=False, handle_oov=True):
        words_found = 0
        result = np.zeros(self.model.wv.vector_size, np.float32)
        for word in ngram_words:
            try:
                word_vector = self.get_vector_word(word=word, use_norm=use_norm_ngram_char, handle_oov=False)
                if use_norm_word:
                    word_vector = self.normalize_vector(word_vector)
                result += word_vector
                words_found += 1
            except KeyError:
                pass
        if not handle_oov and words_found == 0:
            raise KeyError('all words are absent from model')
        else:
            return result / max(1, words_found)


class EmbeddingEpochCallback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{}/{}...".format(self.epoch, model.epochs), end='\r')
        self.epoch += 1

    def on_train_end(self, model):
        print("")


if __name__ == '__main__':
    document = [["the", "quick", "brown", "fox"], ["jumps", "over", "a", "lazy", "dog"]]
    test_word = "a"
    test_sentence = ["there", "is", "a", "test", "duck"]
    embedding = Embedding(min_count=1, size=5, min_n=3)
    embedding.build_vocab(document)
    embedding.train(document, verbose=True)
    print(embedding.get_vector_word(test_word))
    print(embedding.get_vector_sentence(test_sentence, min_n=1, max_n=3))
