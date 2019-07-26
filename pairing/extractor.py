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

    def extract_data(self, data, additional_feature=None, progress_bar=True, with_target=True, as_dataframe=True):
        """
        dict/json list of sentences to tabular data
        """
        result = []
        if progress_bar:
            data = tqdm(data, desc="Extracting data")
        for idx, sentence in enumerate(data):
            additional_feature_sentence = {'id_sentence': idx}
            if additional_feature is not None:
                additional_feature_sentence.update(additional_feature)
            result += self.extract_sentence(sentence=sentence,
                                            additional_feature=additional_feature_sentence,
                                            with_target=with_target,
                                            as_dataframe=False)

        if as_dataframe:
            return pd.DataFrame(result)
        return result

    def extract_sentence(self, sentence, additional_feature=None, with_target=True, as_dataframe=True):
        """
        dict/json of single sentences to tabular data
        """
        result = []
        for aspect_idx, aspect in enumerate(sentence['aspect']):
            for sentiment in sentence['sentiment']:
                result_element = self._extract_features_aspect_sentiment(aspect, sentiment, sentence['token'])
                if additional_feature is not None:
                    result_element.update(additional_feature)
                if with_target:
                    result_element['target'] = self._extract_class_aspect_sentiment(aspect_idx, sentiment)
                result.append(result_element)

        if as_dataframe:
            return pd.DataFrame(result)
        return result

    @staticmethod
    def _extract_class_aspect_sentiment(aspect_idx, sentiment):
        return 1 if aspect_idx in sentiment['index_aspect'] else 0

    def _extract_features_aspect_sentiment(self, aspect, sentiment, tokens):
        result = dict()

        # Intermediate Values
        dictionary_tf = self._extract_feature_dict_tf(tokens)
        if self.embedding_model is not None:
            aspect_embeddings = [self._extract_featre_vector_word(token)
                                 for token in tokens[aspect['start']:aspect['start']+aspect['length']]]
            sentiment_embeddings = [self._extract_featre_vector_word(token)
                                    for token in tokens[sentiment['start']:sentiment['start'] + sentiment['length']]]
            sentence_embedding = self._extract_feature_vector_sentence(tokens)

            if not np.any(aspect_embeddings) or not np.any(sentiment_embeddings):
                min_distance_aspect_sentiment = np.nan
                aspect_chosen_embedding = self._extract_feature_vector_sequence(aspect['start'], aspect['length'], tokens)
                sentiment_chosen_embedding = self._extract_feature_vector_sequence(sentiment['start'], sentiment['length'], tokens)
            else:
                aspect_embeddings = [aspect_embedding for aspect_embedding in aspect_embeddings if np.any(aspect_embedding)]
                sentiment_embeddings = [sentiment_embedding for sentiment_embedding in sentiment_embeddings if np.any(sentiment_embedding)]
                min_distance_aspect_sentiment = np.inf
                aspect_chosen_embedding = None
                sentiment_chosen_embedding = None
                for aspect_embedding in aspect_embeddings:
                    for sentiment_embedding in sentiment_embeddings:
                        distance_aspect_sentiment = self._extract_feature_cosine_distance(aspect_embedding, sentiment_embedding)
                        if distance_aspect_sentiment < min_distance_aspect_sentiment:
                            min_distance_aspect_sentiment = distance_aspect_sentiment
                            aspect_chosen_embedding = aspect_embedding
                            sentiment_chosen_embedding = sentiment_embedding

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
        if self.embedding_model is not None:
            for i in range(len(aspect_chosen_embedding)):
                result['v_aspect_{}'.format(i)] = aspect_chosen_embedding[i]
            for i in range(len(sentiment_chosen_embedding)):
                result['v_sentiment_{}'.format(i)] = sentiment_chosen_embedding[i]
            for i in range(len(sentence_embedding)):
                result['v_sentence_{}'.format(i)] = sentence_embedding[i]

        # Similarity Feature
        if self.embedding_model is not None:
            if np.isnan(min_distance_aspect_sentiment):
                result['cos_aspect_sentiment_validity'] = 0
                result['cos_aspect_sentiment'] = -1
            else:
                result['cos_aspect_sentiment_validity'] = 1
                result['cos_aspect_sentiment'] = min_distance_aspect_sentiment
            result['cos_aspect_sentence'] = self._extract_feature_cosine_distance(aspect_chosen_embedding, sentence_embedding)
            result['cos_sentiment_sentence'] = self._extract_feature_cosine_distance(sentiment_chosen_embedding, sentence_embedding)

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
        return self.embedding_model.get_vector_sentence(sentence=tokens[start:start+length], min_n=length, max_n=length)

    def _extract_feature_vector_sentence(self, tokens):
        return self.embedding_model.get_vector_sentence(sentence=tokens)

    def _extract_featre_vector_word(self, token):
        return self.embedding_model.get_vector_word(word=token)

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
    print(extracted.columns.values)
