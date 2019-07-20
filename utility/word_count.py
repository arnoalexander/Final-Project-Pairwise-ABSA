"""
Calculating scores related to word count
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


class WordCount:

    """
    Utility class
    """

    def __init__(self, saved_filename=None):
        """
        Constructor with optional loading
        """
        if saved_filename is None:
            self.len_data = None
            self.count_dict = None
            self.idf_dict = None
        else:
            self.load(saved_filename)

    def save(self, path):
        """
        Save model to a file
        """
        with open(path, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile)

    def load(self, path):
        """
        Load model from a file
        """
        with open(path, 'rb') as infile:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(infile))

    def fit(self, data, progress_bar=False):
        """
        Fit data (list of list of token) to model
        """
        self.len_data = len(data)
        self.count_dict = self.calculate_count_dict(data=data, progress_bar=progress_bar)
        self.idf_dict = self.calculate_idf_dict(data=data, progress_bar=progress_bar)

    def get_count(self, token):
        """
        Get count of token in train data
        """
        return self.count_dict.get(token, 0)

    def get_idf(self, token):
        """
        Get idf of token based on training data
        """
        return self.idf_dict.get(token, np.log(self.len_data))

    @staticmethod
    def calculate_count_dict(data, progress_bar=False):
        """
        Calculate word count dict from data (list of list of token)
        """
        if progress_bar:
            data = tqdm(data, desc="Calculating count dict")
        count_dict = {}
        for sentence in data:
            for token in sentence:
                count_dict[token] = count_dict.get(token, 0) + 1
        return count_dict

    @staticmethod
    def calculate_idf_dict(data, progress_bar=False):
        """
        Calculate idf dict from data (list of list of token)
        """
        count_sentence = len(data)
        if progress_bar:
            data = tqdm(data, desc="Calculating idf dict")
        idf_dict = {}
        for sentence in data:
            for token in pd.unique(sentence):
                idf_dict[token] = idf_dict.get(token, 0) + 1
        for token, value in idf_dict.items():
            idf_dict[token] = np.log(count_sentence / (value + 1))
        return idf_dict

    @staticmethod
    def calculate_tf_dict(sentence):
        """
        Calculate tf dict from sentence (list of token)
        """
        count_token = len(sentence)
        tf_dict = {}
        for token in sentence:
            tf_dict[token] = tf_dict.get(token, 0) + 1
        for token, value in tf_dict.items():
            tf_dict[token] = value / count_token
        return tf_dict


if __name__ == '__main__':
    data = [["this", "is", "good"], ["hello", "hello", "world"], ["hi"]]
    wc = WordCount()
    wc.fit(data)
    print(wc.count_dict)
    print(wc.idf_dict)
    print(wc.calculate_tf_dict(data[1]))
