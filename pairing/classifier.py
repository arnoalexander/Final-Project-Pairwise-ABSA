"""
Model for classification task
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, confusion_matrix
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, *args, **kwargs):
        self.model = LGBMClassifier(*args, **kwargs)

    def save(self, path):
        with open(path, 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def load(self, path):
        with open(path, 'rb') as infile:
            self.model = pickle.load(infile)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.f1_score(y, self.predict(X), sample_weight=sample_weight)

    @staticmethod
    def f1_score(*args, **kwargs):
        return f1_score(*args, **kwargs)

    @staticmethod
    def confusion_matrix(*args, **kwargs):
        return confusion_matrix(*args, **kwargs)

    @classmethod
    def generate_confusion_matrix_table(cls, y_true, y_pred, labels=None, **kwargs):
        if labels is None:
            labels = np.unique(np.concatenate((np.array(y_true), np.array(y_pred))))
        conf_matrix = cls.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, **kwargs)
        return pd.DataFrame(data=conf_matrix,
                            index=["true_{}".format(label) for label in labels],
                            columns=["predicted_{}".format(label) for label in labels])


if __name__ == '__main__':
    print(Classifier.generate_confusion_matrix_table([0, 0, 0], [0, 1, 1]))
