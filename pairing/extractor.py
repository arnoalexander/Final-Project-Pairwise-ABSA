"""
Extract features from dict/json data containing token, label (aspect and sentiment), and aspect-sentiment pair.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import pandas as pd
from pairing.reader import Reader
import definition


class Extractor:

    """
    Extractor utility class
    """

    @classmethod
    def extract_features_data(cls, data):
        """
        dict/json list of sentences to tabular data
        """
        result = []
        for sentence in data:
            result += cls.extract_features_sentence(sentence)
        return pd.DataFrame(result)

    @classmethod
    def extract_features_sentence(cls, sentence):
        """
        dict/json of single sentences to tabular data
        """
        result = []
        for aspect in sentence['aspect']:
            for sentiment in sentence['sentiment']:
                result.append(cls._extract_features_aspect_sentiment(aspect, sentiment, sentence['token']))
        return result

    @classmethod
    def _extract_features_aspect_sentiment(cls, aspect, sentiment, tokens):
        # TODO implement this with real features
        return {'aspect': tokens[aspect['start']],
                'sentiment': tokens[sentiment['start']],
                'target': 0}


if __name__ == '__main__':
    data = Reader.read_file(definition.DATA_PAIRED_SMALLSAMPLE)
    print(Extractor.extract_features_data(data))
