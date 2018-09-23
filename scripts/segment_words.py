from __future__ import print_function
from __future__ import division

import sys
sys.path.append('../')

import pickle
import pandas as pd
import numpy as np

from env import *
from utils.dataset import DataSet
from utils.transform import *

WORD_INDEX_PKL = os.path.join(CACHES_PATH, 'word_index.pkl')
TRAIN_SEQS_PKL = os.path.join(CACHES_PATH, 'train_seqs.pkl')
VAL_SEQS_PKL = os.path.join(CACHES_PATH, 'val_seqs.pkl')
TEST_SEQS_PKL = os.path.join(CACHES_PATH, 'test_seqs.pkl')
TRAIN_SEQS_PADDED_PKL = os.path.join(CACHES_PATH, 'train_seqs_padded.pkl')
VAL_SEQS_PADDED_PKL = os.path.join(CACHES_PATH, 'val_seqs_padded.pkl')
TEST_SEQS_PADDED_PKL = os.path.join(CACHES_PATH, 'test_seqs_padded.pkl')

dataset = DataSet()
train, val, test = dataset.train, dataset.val, dataset.test

stopwords = gen_stopwords(HIT_TXT)

train_words = list(segment_words(train.content, stopwords))
val_words = list(segment_words(val.content, stopwords))
test_words = list(segment_words(test.content, stopwords))

(train_seqs, val_seqs, test_seqs), word_index = text2seqs(train_words,val_words, test_words)
(train_seqs_padded, val_seqs_padded, test_seqs_padded), maxlen = padseqs(train_seqs, val_seqs, test_seqs)

columns = ['id', 'word_seq']
train_seqs_df = pd.DataFrame(list(zip(train.id, train_seqs)), columns = columns)
val_seqs_df = pd.DataFrame(list(zip(val.id, val_seqs)), columns = columns)
test_seqs_df = pd.DataFrame(list(zip(test.id, test_seqs)), columns = columns)

train_seqs_padded_df = pd.DataFrame(list(zip(train.id, train_seqs_padded)), columns = columns)
val_seqs_padded_df = pd.DataFrame(list(zip(val.id, val_seqs_padded)), columns = columns)
test_seqs_padded_df = pd.DataFrame(list(zip(test.id, test_seqs_padded)), columns = columns)

with open(WORD_INDEX_PKL, 'wb') as f:
    pickle.dump(word_index, f)

with open(TRAIN_SEQS_PKL, 'wb') as f:
    pickle.dump(train_seqs_df, f)

with open(VAL_SEQS_PKL, 'wb') as f:
    pickle.dump(val_seqs_df, f)

with open(TEST_SEQS_PKL, 'wb') as f:
    pickle.dump(test_seqs_df, f)

with open(TRAIN_SEQS_PADDED_PKL, 'wb') as f:
    pickle.dump(train_seqs_padded_df, f)

with open(VAL_SEQS_PADDED_PKL, 'wb') as f:
    pickle.dump(val_seqs_padded_df, f)

with open(TEST_SEQS_PADDED_PKL, 'wb') as f:
    pickle.dump(test_seqs_padded_df, f)