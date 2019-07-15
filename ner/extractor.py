"""
Extract features and class from every possible aspect-sentiment pair from dict/json data
containing token, label (aspect and sentiment), and aspect-sentiment pairing index.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pairing import Reader
from embedding import Embedding
import definition


class Extractor:

    """
    Extractor class for pairing task
    """

    def __init__(self, embedding_filename=None, embedding_model=None, encoder_filename=None, encoder_model=None):
        """
        Initialize object.
        Set embedding_filename (path to saved base model of Embedding class) OR embedding_model (initialized Embedding
        object). If both are specified, embedding_model is preferred. If none, there won't be embedding feature.
        Set encoder_filename (path to saved encoder model) OR encoder_model (initialized encoder model).
        If both are specified, encoder_model is preferred. If none, one will be generated upon data extraction process.
        """
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_filename is not None:
            self.embedding_model = Embedding()
            self.embedding_model.load(embedding_filename)
        else:
            self.embedding_model = None

        if encoder_model is not None:
            self.encoder_model = encoder_model
        elif encoder_filename is not None:
            with open(encoder_filename, 'rb') as infile:
                self.encoder_model = pickle.load(infile)
        else:
            self.encoder_model = None

    def set_encoder(self, encoder_model):
        """
        Set encoder_model with initialized one
        """
        self.encoder_model = encoder_model

    def save_encoder(self, path):
        """
        Save encoder to a file
        """
        if self.encoder_model is not None:
            with open(path, 'wb') as outfile:
                pickle.dump(self.encoder_model, outfile)

    def load_encoder(self, path):
        """
        Load encoder from a file
        """
        with open(path, 'rb') as infile:
            self.encoder_model = pickle.load(infile)

    def fit_data(self, data, progress_bar=True, thresh_count=None):
        """
        Fit data to encoder
        """
        description, _ = self._extract_data_to_json(data=data, progress_bar=progress_bar, with_target=False)
        self._fit_encoder(description=description, thresh_count=thresh_count)

    def extract_data(self, data, progress_bar=True, with_target=True, format_json=False, thresh_count=None):
        """
        dict/json list of sentences to tabular data
        """
        description, target = self._extract_data_to_json(data=data, progress_bar=progress_bar, with_target=with_target)
        if not format_json:
            description = self._extract_json_to_tabular(description=description, thresh_count=thresh_count)
        if with_target:
            return description, target
        return description

    def extract_sentence(self, sentence, with_target=True):
        """
        dict/json of single sentences (list of token) to tabular data
        """
        description = []
        for idx_token in range(len(sentence['token'])):
            description.append(self._extract_features_token(tokens=sentence['token'], idx_token=idx_token))

        if with_target:
            return description, sentence['label']
        return description

    def _extract_data_to_json(self, data, progress_bar=True, with_target=True):
        description = []
        target = []

        if progress_bar:
            data = tqdm(data, desc="Extracting data")
        for sentence in data:
            sentence_extract = self.extract_sentence(sentence=sentence, with_target=with_target)
            if not with_target:
                description.append(sentence_extract)
            else:
                description.append(sentence_extract[0])
                target.append(sentence_extract[1])

        return description, target

    def _extract_json_to_tabular(self, description, thresh_count=None):
        if self.encoder_model is None:
            print("Encoding model not found. It will be generated.")
            self._fit_encoder(description=description, thresh_count=thresh_count)

        values = []
        for sentence in description:
            values_numerical_sentence = []
            values_categorical_sentence = []
            for token in sentence:
                values_numerical_sentence.append([token[key] for key in self.encoder_model['keys_numerical']])
                values_categorical_sentence.append([token[key] for key in self.encoder_model['keys_categorical']])
            values_numerical_sentence = np.vstack(values_numerical_sentence)
            values_categorical_sentence = np.vstack(values_categorical_sentence)
            values_categorical_sentence = self.encoder_model['one_hot_encoder'].transform(values_categorical_sentence).toarray()
            values_sentence = np.hstack((values_numerical_sentence, values_categorical_sentence))
            values.append(values_sentence)

        return np.array(values)

    def _fit_encoder(self, description, thresh_count=None):
        keys_numerical = []
        keys_categorical = []
        keys = [keyvalue[0] for keyvalue in sorted(description[0][0].items())]
        for key in keys:
            if isinstance(description[0][0][key], str):
                keys_categorical.append(key)
            else:
                keys_numerical.append(key)

        values_categorical = []
        for sentence in description:
            for token in sentence:
                values_categorical.append([token[key] for key in keys_categorical])
        values_categorical = np.vstack(values_categorical)
        if thresh_count is None:
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        else:
            unique_count = [np.unique([element[i] for element in values_categorical], return_counts=True)
                            for i in range(np.array(values_categorical).shape[1])]
            constrained_unique_count = [element[0][element[1] >= thresh_count] for element in unique_count]
            one_hot_encoder = OneHotEncoder(categories=constrained_unique_count, handle_unknown='ignore')
        one_hot_encoder.fit(values_categorical)

        self.encoder_model = {'keys_numerical': keys_numerical,
                              'keys_categorical': keys_categorical,
                              'one_hot_encoder': one_hot_encoder}

    def _extract_features_token(self, tokens, idx_token):
        # TODO implement more feature
        result = dict()
        result['token'] = tokens[idx_token]
        result['position'] = idx_token
        return result


if __name__ == '__main__':
    data = Reader.read_file(os.path.join(definition.DATA_LABELLED, 'sample-small.txt'))
    extractor = Extractor()
    description, label = extractor.extract_data(data, thresh_count=2)
    print(extractor.encoder_model['one_hot_encoder'].categories_)
    print(description[0])
    print(label[0])
