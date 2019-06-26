# definition of important project paths

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, 'data')
DATA_LABELLED = os.path.join(DATA, 'labelled')
DATA_RAW = os.path.join(DATA, 'raw')
DATA_LABELLED_TRAIN = os.path.join(DATA_LABELLED, 'train')
DATA_LABELLED_TEST = os.path.join(DATA_LABELLED, 'test')
DATA_RAW_FILE = os.path.join(DATA_RAW, 'ceres_review_until_20181201.csv')
