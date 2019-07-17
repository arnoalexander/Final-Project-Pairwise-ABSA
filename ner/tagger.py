"""
Architectures for sequence tagging task
"""

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_contrib.utils import save_load_utils
from keras.models import Model, load_model
from keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense
from keras.utils import Sequence
import numpy as np


class BaseTagger:

    """
    Base for all tagger model
    """

    def __init__(self, model):
        self.model = model

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self.model.fit_generator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self.model.predict_generator(*args, **kwargs)

    def save(self, filepath):
        self.model.save(filepath=filepath, include_optimizer=False)

    def load(self, filepath):
        self.model = load_model(filepath=filepath, custom_objects=self.create_custom_objects())

    @classmethod
    def create_custom_objects(cls):
        instance_holder = {"instance": None}

        class ClassWrapper(CRF):
            def __init__(self, *args, **kwargs):
                instance_holder["instance"] = self
                super(ClassWrapper, self).__init__(*args, **kwargs)

        def loss(*args):
            method = getattr(instance_holder["instance"], "loss_function")
            return method(*args)

        def accuracy(*args):
            method = getattr(instance_holder["instance"], "accuracy")
            return method(*args)

        return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "loss": loss, "accuracy": accuracy}


class BiLstmCrfTagger(BaseTagger):

    """
    Bi-directional LSTM + CRF Architecture
    """

    def __init__(self, n_features, n_lstm_unit, n_distributed_dense, n_tags):
        inputs = Input(shape=(None, n_features))
        outputs = Bidirectional(LSTM(units=n_lstm_unit, return_sequences=True))(inputs)
        outputs = TimeDistributed(Dense(n_distributed_dense))(outputs)
        outputs = CRF(n_tags)(outputs)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_accuracy])
        super().__init__(self.model)


class UnaryBatchGenerator(Sequence):

    """
    Generate batch with size=1 for training
    """

    def __init__(self, X, y=None, shuffle=False):
        """
        Constructor
        """
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        """
        Number of batch per epoch
        """
        return len(self.X)

    def __getitem__(self, batch_id):
        """
        Get batch_id th batch
        """
        if self.y is None:
            return np.array([self.X[batch_id]])
        return np.array([self.X[batch_id]]), np.array([self.y[batch_id]])

    def on_epoch_end(self):
        """
        Shuffles indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.X = self.X[self.indexes]
            if self.y is not None:
                self.y = self.y[self.indexes]


if __name__ == '__main__':
    model = BiLstmCrfTagger(3, 50, 20, 2)
    sample_data = np.array([[[1.0, 2.0, 0.0], [0.5, 1.2, 0.2]], [[1.0, 2.0, 0.1]]])
    sample_label = np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0]]])
    generator = UnaryBatchGenerator(sample_data, sample_label)
    model.fit_generator(generator, epochs=5, verbose=2)
    pred = []
    for item in sample_data:
        pred.append((model.predict(np.array([item]))[0]))
    print(pred)
    print(model.summary())
