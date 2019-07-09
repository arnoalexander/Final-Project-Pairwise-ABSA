"""
Extract features and class from every possible aspect-sentiment pair from dict/json data
containing token, label (aspect and sentiment), and aspect-sentiment pairing index.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import pandas as pd
from tqdm import tqdm
from pairing.reader import Reader
import definition


class Extractor:

    """
    Extractor utility class
    """

    @classmethod
    def extract_data(cls, data, progress_bar=True):
        """
        dict/json list of sentences to tabular data
        """
        result = []
        if progress_bar:
            data = tqdm(data)
        for sentence in data:
            result += cls.extract_sentence(sentence)
        return pd.DataFrame(result)

    @classmethod
    def extract_sentence(cls, sentence):
        """
        dict/json of single sentences to tabular data
        """
        result = []
        for aspect_idx, aspect in enumerate(sentence['aspect']):
            for sentiment in sentence['sentiment']:
                result_element = cls._extract_features_aspect_sentiment(aspect, sentiment, sentence['token'])
                result_element['target'] = cls._extract_class_aspect_sentiment(aspect_idx, sentiment)
                result.append(result_element)
        return result

    @classmethod
    def _extract_class_aspect_sentiment(cls, aspect_idx, sentiment):
        return 1 if aspect_idx in sentiment['index_aspect'] else 0

    @classmethod
    def _extract_features_aspect_sentiment(cls, aspect, sentiment, tokens):
        result = dict()

        # TODO remove this upon real implementation. only for test purpose
        result['aspect'] = tokens[aspect['start']]
        result['sentiment'] = tokens[sentiment['start']]

        # TODO add more feature
        # Statistics Feature
        result['len_aspect_word'] = aspect['length']
        result['len_sentiment_word'] = sentiment['length']
        result['len_aspect_char'] = cls._extract_feature_len_sequence_char(aspect['start'], aspect['length'], tokens)
        result['len_sentiment_char'] = cls._extract_feature_len_sequence_char(sentiment['start'], sentiment['length'], tokens)
        result['position_aspect'] = aspect['start']
        result['position_sentiment'] = sentiment['start']
        result['dist_start'] = abs(aspect['start'] - sentiment['start'])
        result['dist_endpoint'] = cls._extract_feature_dist_endpoint(aspect, sentiment)

        return result

    @classmethod
    def _extract_feature_len_sequence_char(cls, start, length, tokens):
        return sum([len(token) for token in tokens[start:start+length]])

    @classmethod
    def _extract_feature_dist_endpoint(cls, aspect, sentiment):
        if aspect['start'] < sentiment['start']:
            return sentiment['start'] - (aspect['start'] + aspect['length'] - 1)
        return aspect['start'] - (sentiment['start'] + sentiment['length'] - 1)


if __name__ == '__main__':
    data = Reader.read_file(definition.DATA_PAIRED_SMALLSAMPLE)
    print(Extractor.extract_data(data))
