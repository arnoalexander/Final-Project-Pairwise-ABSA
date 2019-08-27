"""
Model for classification task
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, confusion_matrix
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle


class BaseClassifier (BaseEstimator, ClassifierMixin):

    """
    Base class for all classifier
    """

    def fit(self, X, y, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        pass

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


class BaselineClassifier(BaseClassifier):

    """
    Pair aspect with closest sentiment
    """

    def predict(self, X, *args, **kwargs):
        """
        Predict label for X. X is dataframe with mandatory column : _id_sentiment & _id_closest_sentiment
        """
        return (X['_id_sentiment'] == X['_id_closest_sentiment']).to_numpy().astype(int)


class GBClassifier(BaseClassifier):

    """
    Gradient boosting classifier
    """

    def __init__(self, *args, model_base=None, model_filename=None, **kwargs):
        """
        Initialize object.
        Set model_filename (path to saved base classifier model) OR model_base(initialized classifier model).
        If both are specified, model_base is preferred. If none, new model will be generated based on args and kwargs.
        """
        if model_base is not None:
            self.model = model_base
        elif model_filename is not None:
            self.load(path=model_filename)
        else:
            self.model = LGBMClassifier(*args, **kwargs)

    def save(self, path):
        """
        Save base model to a file
        """
        with open(path, 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def load(self, path):
        """
        Load base model from a file
        """
        with open(path, 'rb') as infile:
            self.model = pickle.load(infile)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit data X with label y
        """
        self.model.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """
        Predict label for X
        """
        return self.model.predict(X, *args, **kwargs)


class FilteredGBClassifier(GBClassifier):

    """
    Gradient boosting classifier with obvious-case bypassing
    """

    def __init__(self, *args, model_base=None, model_filename=None, **kwargs):
        """
        Initialize object.
        Set model_filename (path to saved base classifier model) OR model_base(initialized classifier model).
        If both are specified, model_base is preferred. If none, new model will be generated based on args and kwargs.
        """
        super().__init__(*args, model_base=model_base, model_filename=model_filename, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit data X with label y. X is dataframe with mandatory dummy column : _n_sentiment
        """
        train_criteria = self._get_train_criteria(X)
        X_train = X[train_criteria].drop(labels=['_n_sentiment'], axis=1)
        y_train = y[train_criteria]
        if len(y_train) > 0:
            self.model.fit(X_train, y_train, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """
        Predict label for X. X is dataframe with mandatory dummy column : _n_sentiment
        """
        train_criteria = self._get_train_criteria(X)
        X_pass = X[np.bitwise_not(train_criteria)]
        X_train = X[train_criteria].drop(labels=['_n_sentiment'], axis=1)
        pred_pass = pd.Series(data=np.repeat(a=1, repeats=len(X_pass)), index=X_pass.index)
        pred_train = pd.Series(data=self.model.predict(X_train, *args, **kwargs), index=X_train.index)
        pred = pd.concat((pred_pass, pred_train)).sort_index().as_matrix()
        return pred

    @staticmethod
    def _get_train_criteria(X):
        return X['_n_sentiment'] > 1


if __name__ == '__main__':
    print(GBClassifier.generate_confusion_matrix_table([0, 0, 0], [0, 1, 1]))
