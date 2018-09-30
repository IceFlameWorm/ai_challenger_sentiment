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

SAVED_PATH = os.path.join(CACHES_PATH, 'word_seqs/simple')

####### Temp ######
CHARS_SAVED_PATH = os.path.join(CACHES_PATH, 'char_seqs/simple')
TRAIN_CHARS_PKL = os.path.join(CHARS_SAVED_PATH, 'train_chars.pkl')
VAL_CHARS_PKL = os.path.join(CHARS_SAVED_PATH, 'val_chars.pkl')
TEST_CHARS_PKL = os.path.join(CHARS_SAVED_PATH, 'test_chars.pkl')
###################

TRAIN_WORDS_PKL = os.path.join(SAVED_PATH, 'train_words.pkl')
VAL_WORDS_PKL = os.path.join(SAVED_PATH, 'val_words.pkl')
TEST_WORDS_PKL = os.path.join(SAVED_PATH, 'test_words.pkl')
WORD_INDEX_PKL = os.path.join(SAVED_PATH, 'word_index.pkl')
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

train_simple = trad2simple(train.content.apply(rm_quomarks))
val_simple = trad2simple(val.content.apply(rm_quomarks))
test_simple = trad2simple(test.content.apply(rm_quomarks))

# rm non-Chinese chars
train_simple = rm_non_Chinese(train_simple)
val_simple = rm_non_Chinese(val_simple)
test_simple = rm_non_Chinese(test_simple)

####### Temp ######
train_chars_ser = train_simple.apply(list)
val_chars_ser = val_simple.apply(list)
test_chars_ser = test_simple.apply(list)
###################

train_words_ser = segment_words(train_simple, stopwords)
val_words_ser = segment_words(val_simple, stopwords)
test_words_ser = segment_words(test_simple, stopwords)

train_words = list(train_words_ser)
val_words = list(val_words_ser)
test_words = list(test_words_ser)

(train_seqs, val_seqs, test_seqs), word_index = text2seqs(train_words,val_words, test_words)
(train_seqs_padded, val_seqs_padded, test_seqs_padded), maxlen = padseqs(train_seqs, val_seqs, test_seqs)

####### Temp #####
columns = ['id', 'chars']

train_chars_df = pd.DataFrame(list(zip(train.id, train_chars_ser)), columns = columns)
val_chars_df = pd.DataFrame(list(zip(val.id, val_chars_ser)), columns = columns)
test_chars_df = pd.DataFrame(list(zip(test.id, test_chars_ser)), columns = columns)
##################

columns = ['id', 'words']

train_words_df = pd.DataFrame(list(zip(train.id, train_words_ser)), columns = columns)
val_words_df = pd.DataFrame(list(zip(val.id, val_words_ser)), columns = columns)
test_words_df = pd.DataFrame(list(zip(test.id, test_words_ser)), columns = columns)

columns = ['id', 'words_seq']

train_seqs_df = pd.DataFrame(list(zip(train.id, train_seqs)), columns = columns)
val_seqs_df = pd.DataFrame(list(zip(val.id, val_seqs)), columns = columns)
test_seqs_df = pd.DataFrame(list(zip(test.id, test_seqs)), columns = columns)

train_seqs_padded_df = pd.DataFrame(list(zip(train.id, train_seqs_padded)), columns = columns)
val_seqs_padded_df = pd.DataFrame(list(zip(val.id, val_seqs_padded)), columns = columns)
test_seqs_padded_df = pd.DataFrame(list(zip(test.id, test_seqs_padded)), columns = columns)

####### Temp #####
with open(TRAIN_CHARS_PKL, 'wb') as f:
    pickle.dump(train_chars_df, f)

with open(VAL_CHARS_PKL, 'wb') as f:
    pickle.dump(val_chars_df, f)

with open(TEST_CHARS_PKL, 'wb') as f:
    pickle.dump(test_chars_df, f)
##################

with open(TRAIN_WORDS_PKL, 'wb') as f:
    pickle.dump(train_words_df, f)

with open(VAL_WORDS_PKL, 'wb') as f:
    pickle.dump(val_words_df, f)

with open(TEST_WORDS_PKL, 'wb') as f:
    pickle.dump(test_words_df, f)

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
