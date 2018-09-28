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

SAVED_PATH = os.path.join(CACHES_PATH, 'char_seqs/simple')

CHAR_INDEX_PKL = os.path.join(SAVED_PATH, 'char_index.pkl')
TRAIN_SEQS_PKL = os.path.join(SAVED_PATH, 'train_seqs.pkl')
VAL_SEQS_PKL = os.path.join(SAVED_PATH, 'val_seqs.pkl')
TEST_SEQS_PKL = os.path.join(SAVED_PATH, 'test_seqs.pkl')
TRAIN_SEQS_PADDED_PKL = os.path.join(SAVED_PATH, 'train_seqs_padded.pkl')
VAL_SEQS_PADDED_PKL = os.path.join(SAVED_PATH, 'val_seqs_padded.pkl')
TEST_SEQS_PADDED_PKL = os.path.join(SAVED_PATH, 'test_seqs_padded.pkl')

dataset = DataSet()
train, val, test = dataset.train, dataset.val, dataset.test

stopwords = gen_stopwords(HIT_TXT)

# tradtional 2 simple
def rm_quomarks(x):
    return x[1:-1]

train_simple = list(trad2simple(train.content.apply(rm_quomarks)))
val_simple = list(trad2simple(val.content.apply(rm_quomarks)))
test_simple = list(trad2simple(test.content.apply(rm_quomarks)))

# train_words = list(segment_words(train_simple, stopwords))
# val_words = list(segment_words(val_simple, stopwords))
# test_words = list(segment_words(test_simple, stopwords))

(train_seqs, val_seqs, test_seqs), char_index = text2seqs(train_simple,val_simple, test_simple, char_level = True)
(train_seqs_padded, val_seqs_padded, test_seqs_padded), maxlen = padseqs(train_seqs, val_seqs, test_seqs)

columns = ['id', 'char_seq']
train_seqs_df = pd.DataFrame(list(zip(train.id, train_seqs)), columns = columns)
val_seqs_df = pd.DataFrame(list(zip(val.id, val_seqs)), columns = columns)
test_seqs_df = pd.DataFrame(list(zip(test.id, test_seqs)), columns = columns)

train_seqs_padded_df = pd.DataFrame(list(zip(train.id, train_seqs_padded)), columns = columns)
val_seqs_padded_df = pd.DataFrame(list(zip(val.id, val_seqs_padded)), columns = columns)
test_seqs_padded_df = pd.DataFrame(list(zip(test.id, test_seqs_padded)), columns = columns)

with open(CHAR_INDEX_PKL, 'wb') as f:
    pickle.dump(char_index, f)

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
