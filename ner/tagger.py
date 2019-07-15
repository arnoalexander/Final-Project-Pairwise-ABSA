"""
Architectures for sequence tagging task
"""

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
import numpy as np


class BaseTagger:

    """
    Base for all tagger model
    """

    def __init__(self, model):
        self.model = model

    def summary(self):
        return self.model.summary()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)


class BiLstmCrfTagger(BaseTagger):

    """
    Bi-directional LSTM + CRF Architecture
    """

    def __init__(self, n_features, n_lstm_unit, n_distributed_dense, n_tags):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(input_shape=(None, n_features), units=n_lstm_unit, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(n_distributed_dense)))
        self.model.add(CRF(n_tags))
        self.model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_accuracy])
        super().__init__(self.model)


if __name__ == '__main__':
    model = BiLstmCrfTagger(3, 50, 20, 2)
    sample_data = np.array([[[1.0, 2.0, 0.0], [0.5, 1.2, 0.2]]])
    sample_label = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    model.fit(sample_data, sample_label, epochs=5)
    print(model.summary())
