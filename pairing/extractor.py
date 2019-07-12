"""
Extract features and class from every possible aspect-sentiment pair from dict/json data
containing token, label (aspect and sentiment), and aspect-sentiment pairing index.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pairing import Reader
from embedding import Embedding
import definition


class Extractor:

    """
    Extractor class
    """

    def __init__(self, embedding_filename=None, embedding_model=None):
        """
        Initialize object.
        Set embedding_filename (path to saved base model of Embedding class) OR embedding model (initialized Embedding
        object. If both are specified, embedding_model is preferred. If none, there won't be embedding feature
        """
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_filename is not None:
            self.embedding_model = Embedding()
            self.embedding_model.load(embedding_filename)
        else:
            self.embedding_model = None

    def extract_data(self, data, progress_bar=True):
        """
        dict/json list of sentences to tabular data
        """
        result = []
        if progress_bar:
            data = tqdm(data)
        for sentence in data:
            result += self.extract_sentence(sentence)
        return pd.DataFrame(result)

    def extract_sentence(self, sentence):
        """
        dict/json of single sentences to tabular data
        """
        result = []
        for aspect_idx, aspect in enumerate(sentence['aspect']):
            for sentiment in sentence['sentiment']:
                result_element = self._extract_features_aspect_sentiment(aspect, sentiment, sentence['token'])
                result_element['target'] = self._extract_class_aspect_sentiment(aspect_idx, sentiment)
                result.append(result_element)
        return result

    @staticmethod
    def _extract_class_aspect_sentiment(aspect_idx, sentiment):
        return 1 if aspect_idx in sentiment['index_aspect'] else 0

    def _extract_features_aspect_sentiment(self, aspect, sentiment, tokens):
        result = dict()

        # Intermediate Values
        aspect_embedding = self._extract_feature_vector_sequence(aspect['start'], aspect['length'], tokens)
        sentiment_embedding = self._extract_feature_vector_sequence(sentiment['start'], sentiment['length'], tokens)
        sentence_embedding = self._extract_feature_vector_sentence(tokens)

        # TODO remove this upon real implementation. only for test purpose
        # result['aspect'] = tokens[aspect['start']]
        # result['sentiment'] = tokens[sentiment['start']]

        # TODO add more feature
        # Statistics Feature
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
    def _extract_feature_cosine_distance(vector1, vector2):
        return Embedding.cosine_distance(vector1, vector2)


if __name__ == '__main__':
    data = Reader.read_file(definition.DATA_PAIRED_SMALLSAMPLE)
    extractor = Extractor()
    print(extractor.extract_data(data))
