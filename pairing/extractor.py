"""
Extract features and class from every possible aspect-sentiment pair from dict/json data
containing token, label (aspect and sentiment), and aspect-sentiment pairing index.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pairing import Reader
from utility import Embedding, WordCount
import definition


class Extractor:

    """
    Extractor class for pairing task
    """

    def __init__(self, embedding_filename=None, embedding_model=None, word_count_filename=None, word_count_model=None):
        """
        Initialize object.
        Set filename (path to saved model) OR model (initialized model). If both are specified, model is preferred.
        If none, there won't be feature related to model.
        """
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_filename is not None:
            self.embedding_model = Embedding()
            self.embedding_model.load(path=embedding_filename)
        else:
            self.embedding_model = None

        if word_count_model is not None:
            self.word_count_model = word_count_model
        elif word_count_filename is not None:
            self.word_count_model = WordCount()
            self.word_count_model.load(path=word_count_filename)
        else:
            self.word_count_model = None

    def set_embedding_model(self, embedding_model):
        """
        Set embedding model with initialized Embedding object
        """
        self.embedding_model = embedding_model

    def load_embedding_model(self, path):
        """
        Load embedding model from a file (using Embedding load method)
        """
        self.embedding_model = Embedding()
        self.embedding_model.load(path=path)

    def set_word_count_model(self, word_count_model):
        """
        Set word count model with initialized WordCount object
        """
        self.word_count_model = word_count_model

    def load_word_count_model(self, path):
        """
        Load word count model from a file (using WordCount load method)
        """
        self.word_count_model = WordCount()
        self.word_count_model.load(path=path)

    def extract_data(self, data, progress_bar=True, with_target=True):
        """
        dict/json list of sentences to tabular data
        """
        result = []
        if progress_bar:
            data = tqdm(data, desc="Extracting data")
        for sentence in data:
            result += self.extract_sentence(sentence=sentence, with_target=with_target)
        return pd.DataFrame(result)

    def extract_sentence(self, sentence, with_target=True):
        """
        dict/json of single sentences to tabular data
        """
        result = []
        for aspect_idx, aspect in enumerate(sentence['aspect']):
            for sentiment in sentence['sentiment']:
                result_element = self._extract_features_aspect_sentiment(aspect, sentiment, sentence['token'])
                if with_target:
                    result_element['target'] = self._extract_class_aspect_sentiment(aspect_idx, sentiment)
                result.append(result_element)
        return result

    @staticmethod
    def _extract_class_aspect_sentiment(aspect_idx, sentiment):
        return 1 if aspect_idx in sentiment['index_aspect'] else 0

    def _extract_features_aspect_sentiment(self, aspect, sentiment, tokens):
        result = dict()

        # Intermediate Variables
        dictionary_tf = self._extract_feature_dict_tf(tokens)
        aspect_embedding = self._extract_feature_vector_sequence(aspect['start'], aspect['length'], tokens)
        sentiment_embedding = self._extract_feature_vector_sequence(sentiment['start'], sentiment['length'], tokens)
        sentence_embedding = self._extract_feature_vector_sentence(tokens)

        # TODO remove this upon real implementation. only for test purpose
        # result['aspect'] = tokens[aspect['start']]
        # result['sentiment'] = tokens[sentiment['start']]

        # Statistics Feature
        if self.word_count_model is not None:
            result['idf_aspect'] = self._extract_feature_idf(aspect['start'], aspect['length'], tokens)
            result['idf_sentiment'] = self._extract_feature_idf(sentiment['start'], sentiment['length'], tokens)
        result['tf_aspect'] = self._extract_feature_tf(dictionary_tf, aspect['start'], aspect['length'], tokens)
        result['tf_sentiment'] = self._extract_feature_tf(dictionary_tf, sentiment['start'], sentiment['length'], tokens)
        result['len_aspect_word'] = aspect['length']
        result['len_sentiment_word'] = sentiment['length']
        result['len_aspect_char'] = self._extract_feature_numchar_sequence(aspect['start'], aspect['length'], tokens)
        result['len_sentiment_char'] = self._extract_feature_numchar_sequence(sentiment['start'], sentiment['length'], tokens)
        result['position_aspect'] = aspect['start']
        result['position_sentiment'] = sentiment['start']
        result['dist_start'] = abs(aspect['start'] - sentiment['start'])
        result['dist_endpoint'] = self._extract_feature_dist_endpoint(aspect, sentiment)

        # Semantic Feature
        for i in range(len(aspect_embedding)):
            result['v_aspect_{}'.format(i)] = aspect_embedding[i]
        for i in range(len(sentiment_embedding)):
            result['v_sentiment_{}'.format(i)] = sentiment_embedding[i]
        for i in range(len(sentence_embedding)):
            result['v_sentence_{}'.format(i)] = sentence_embedding[i]

        # Similarity Feature
        if self.embedding_model is not None:
            result['cos_aspect_sentiment'] = self._extract_feature_cosine_distance(aspect_embedding, sentiment_embedding)
            result['cos_aspect_sentence'] = self._extract_feature_cosine_distance(aspect_embedding, sentence_embedding)
            result['cos_sentiment_sentence'] = self._extract_feature_cosine_distance(sentiment_embedding, sentence_embedding)

        return result

    @staticmethod
    def _extract_feature_numchar_sequence(start, length, tokens):
        return sum([len(token) for token in tokens[start:start+length]])

    @staticmethod
    def _extract_feature_dist_endpoint(aspect, sentiment):
        if aspect['start'] < sentiment['start']:
            return sentiment['start'] - (aspect['start'] + aspect['length'] - 1)
        return aspect['start'] - (sentiment['start'] + sentiment['length'] - 1)

    def _extract_feature_vector_sequence(self, start, length, tokens):
        if self.embedding_model is None:
            return np.array([])
        return self.embedding_model.get_vector_sentence(sentence=tokens[start:start+length], min_n=length, max_n=length)

    def _extract_feature_vector_sentence(self, tokens):
        if self.embedding_model is None:
            return np.array([])
        return self.embedding_model.get_vector_sentence(sentence=tokens)

    @staticmethod
    def _extract_feature_dict_tf(tokens):
        return WordCount.calculate_tf_dict(sentence=tokens)

    @staticmethod
    def _extract_feature_tf(dict_tf, start, length, tokens):
        sum_tf = 0
        for index in range(start, start + length):
            sum_tf += dict_tf.get(tokens[index], 0)
        return sum_tf / length

    def _extract_feature_idf(self, start, length, tokens):
        if self.word_count_model is None:
            return 0
        sum_idf = 0
        for index in range(start, start + length):
            sum_idf += self.word_count_model.get_idf(tokens[index])
        return sum_idf / length

    @staticmethod
    def _extract_feature_cosine_distance(vector1, vector2):
        return Embedding.cosine_distance(vector1, vector2)


if __name__ == '__main__':
    data = Reader.read_file(os.path.join(definition.DATA_LABELLED, 'sample-small.txt'))
    extractor = Extractor()
    extractor.load_embedding_model(os.path.join(definition.MODEL_UTILITY, 'fasttext_25.bin'))
    extractor.load_word_count_model(os.path.join(definition.MODEL_UTILITY, 'word_count_60.pkl'))
    extracted = extractor.extract_data(data)
    print(extracted.values.shape)
    print(extracted)
