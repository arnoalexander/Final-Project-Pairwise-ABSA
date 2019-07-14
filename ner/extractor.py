"""
Extract features and class from every possible aspect-sentiment pair from dict/json data
containing token, label (aspect and sentiment), and aspect-sentiment pairing index.
Dict/json format follows Reader's output.

The result is tabular data with defined features
"""

import os
import pickle
from tqdm import tqdm
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
        Set encoder_filename (path to saved encoder dictionary) OR encoder_model (initialized encoder dictionary).
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

    def fit_data(self, data, progress_bar=True):
        """
        Fit data to encoder
        """
        description, _ = self._extract_data_json(data=data, progress_bar=progress_bar, with_target=False, flag_fit=True)
        # TODO fit description to encoder

    def extract_data(self, data, progress_bar=True, with_target=True, format_json=False):
        """
        dict/json list of sentences to tabular data
        """
        description, target = self._extract_data_json(data=data, progress_bar=progress_bar, with_target=with_target)
        if not format_json:
            pass  # TODO process description into tabular
        if with_target:
            return description, target
        return description

    def extract_sentence(self, sentence, with_target=True):
        """
        dict/json of single sentences (list of token) to tabular data
        """
        description = []
        for idx_token in range(len(sentence)):
            description.append(self._extract_features_token(tokens=sentence['token'], idx_token=idx_token))

        if with_target:
            return description, sentence['label']
        return description

    def _extract_data_json(self, data, progress_bar=True, with_target=True, flag_fit=False):
        description = []
        target = []

        if progress_bar:
            if flag_fit:
                data = tqdm(data, desc="Fitting data")
            else:
                data = tqdm(data, desc="Extracting data")
        for sentence in data:
            sentence_extract = self.extract_sentence(sentence=sentence, with_target=with_target)
            if not with_target:
                description += sentence_extract
            else:
                description += sentence_extract[0]
                target += sentence_extract[1]

        return description, target

    def _extract_features_token(self, tokens, idx_token):
        # TODO implement more feature
        return {'token': tokens[idx_token]}


if __name__ == '__main__':
    data = Reader.read_file(os.path.join(definition.DATA_LABELLED, 'sample-small.txt'))
    extractor = Extractor()
    extractor.fit_data(data)
    print(extractor.extract_data(data))
