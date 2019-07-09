# definition of important project paths

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, 'data')
DATA_LABELLED = os.path.join(DATA, 'labelled')
DATA_LABELLED_TRAIN = os.path.join(DATA_LABELLED, 'train.txt')
DATA_LABELLED_TEST = os.path.join(DATA_LABELLED, 'test.txt')
DATA_PAIRED_SMALLSAMPLE = os.path.join(DATA_LABELLED, 'sample-small.txt')
DATA_PAIRED_SAMPLE = os.path.join(DATA_LABELLED, 'sample-pair.txt')
DATA_PAIRED_TRAIN = os.path.join(DATA_LABELLED, 'train-pair.txt')
DATA_RAW = os.path.join(DATA, 'raw')
DATA_RAW_FILE = os.path.join(DATA_RAW, 'ceres_review_until_20181201.csv')

MODEL = os.path.join(ROOT, 'model')
MODEL_EMBEDDING = os.path.join(MODEL, 'embedding')
MODEL_EMBEDDING_FASTTEXT = os.path.join(MODEL_EMBEDDING, 'fasttext.bin')
MODEL_EMBEDDING_SAMPLEFASTTEXT = os.path.join(MODEL_EMBEDDING, 'fasttext_sample.bin')
