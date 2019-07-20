# definition of important project paths

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, 'data')
DATA_LABELLED = os.path.join(DATA, 'labelled')
DATA_LABELLED_TRAIN = os.path.join(DATA_LABELLED, 'train.txt')
DATA_LABELLED_TEST = os.path.join(DATA_LABELLED, 'test.txt')
DATA_PAIRED_SAMPLE = os.path.join(DATA_LABELLED, 'sample-pair.txt')
DATA_PAIRED_TRAIN = os.path.join(DATA_LABELLED, 'train-pair.txt')
DATA_RAW = os.path.join(DATA, 'raw')
DATA_RAW_FILE = os.path.join(DATA_RAW, 'ceres_review_until_20181201.csv')

MODEL = os.path.join(ROOT, 'model')
MODEL_UTILITY = os.path.join(MODEL, 'utility')
MODEL_NER = os.path.join(MODEL, 'ner')
MODEL_ENCODING_FILE = os.path.join(MODEL_NER, 'encoding.pkl')
MODEL_ENCODING_SAMPLEFILE = os.path.join(MODEL_NER, 'encoding_sample.pkl')
MODEL_NER_KERAS = os.path.join(MODEL_NER, 'keras.h5')
MODEL_NER_SAMPLEKERAS = os.path.join(MODEL_NER, 'keras_sample.h5')
MODEL_PAIRING = os.path.join(MODEL, 'pairing')
MODEL_PAIRING_FILE = os.path.join(MODEL_PAIRING, 'pairing.pkl')
MODEL_PAIRING_SAMPLEFILE = os.path.join(MODEL_PAIRING, 'pairing_sample.pkl')
