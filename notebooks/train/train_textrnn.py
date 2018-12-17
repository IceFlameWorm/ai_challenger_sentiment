
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')


# In[2]:


import random
random.seed(4444)
import pickle
import numpy as np
np.random.seed(5555)
import tensorflow as tf
tf.set_random_seed(6666)
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from keras.optimizers import RMSprop, Adam
from utils.dataset import DataSet, LABELS
from models.textrnn import CuDNNGRUSeq as TextRNN
from models.sentiment_base import SCompositeModel as CompositeTextRNN
from env import *


# In[3]:


WORD_SEQS_PATH = os.path.join(CACHES_PATH, 'word_seqs_sw', 'simple')

TRAIN_SEQS_PADDED_PKL = os.path.join(WORD_SEQS_PATH, 'train_seqs_padded.pkl')
VAL_SEQS_PADDED_PKL = os.path.join(WORD_SEQS_PATH, 'val_seqs_padded.pkl')
TEST_SEQS_PADDED_PKL = os.path.join(WORD_SEQS_PATH, 'test_seqs_padded.pkl')

SAVED_PATH = os.path.join(SAVED_MODELS_PATH, 'textrnn', 'cudnngruseq_test')
if not os.path.exists(SAVED_PATH):
    os.makedirs(SAVED_PATH)

MODEL_PRE = 'cudnngruseq_'

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
FACTOR = 0.2

VECTOR_DIM = 300
EMBEDDING_PKL = os.path.join(WORD_SEQS_PATH, 'wem_%d.pkl' % VECTOR_DIM)
RESULT_CSV = os.path.join(RESULTS_PATH, 'cudnngruseq_sw.csv')


# In[4]:


raw_dataset = DataSet()
train, val, test = raw_dataset.train, raw_dataset.val, raw_dataset.test

with open(TRAIN_SEQS_PADDED_PKL, 'rb') as f:
    train_seqs_padded = pickle.load(f)
    
with open(VAL_SEQS_PADDED_PKL, 'rb') as f:
    val_seqs_padded = pickle.load(f)
    
with open(TEST_SEQS_PADDED_PKL, 'rb') as f:
    test_seqs_padded = pickle.load(f)
    
with open(EMBEDDING_PKL, 'rb') as f:
    embedding = pickle.load(f)


# In[5]:


train_with_seq = pd.merge(train, train_seqs_padded, on='id')
val_with_seq = pd.merge(val, val_seqs_padded, on='id')
test_with_seq = pd.merge(test, test_seqs_padded, on='id')


# In[ ]:


y_cols = LABELS

seq = 'words_seq'

train_x = np.array(list(train_with_seq[seq]))
train_y = train_with_seq[y_cols]
val_x = np.array(list(val_with_seq[seq]))
val_y = val_with_seq[y_cols]
test_x = np.array(list(test_with_seq[seq]))

comps = [(TextRNN, {"max_len":train_x.shape[1], 
                    'embedding': embedding,
                    'mask_zero': False})] * len(y_cols)
# comps = [(LinXiSingleModel, {"maxlen":train_x.shape[1]})] * len(y_cols)
comp_model = CompositeTextRNN(comps)
comp_model.fit(train_x, train_y, val = (val_x, val_y), y_cols = y_cols, seq = 'word_seq', saved_path = SAVED_PATH,
               model_pre = MODEL_PRE, lr = LR, epochs = EPOCHS, batch_size = BATCH_SIZE, optimizer = RMSprop,
               factor = FACTOR)


# In[ ]:


MODEL_PRE = os.path.join(SAVED_PATH, MODEL_PRE)
# comps_1 = [(LinXiSingleModel, {'model_file': MODEL_PRE + str(i) + '.h5'}) for i in range(len(y_cols))]
# comp_model_1 = LinXiCompositeModel(comps_1)


# In[ ]:


weights_files = [MODEL_PRE + str(i) + '.h5' for i in range(len(y_cols))]
comp_model.load_weights(weights_files)


# In[ ]:


val_preds = comp_model.predict(val_x)
test_preds = comp_model.predict(test_x)


# In[ ]:


f1s = 0
for i, (vy, vp) in enumerate(zip(val_y.values.T + 2, val_preds)):
    f1 = f1_score(vy, vp, average='macro')
    print("The %sth f1: %s" % (i, f1))
    f1s += f1
    
print("The average f1 of val is %s" % (f1s / len(y_cols)))


# In[ ]:


res = test.copy()
for index, col in enumerate(LABELS):
    res[col] = test_preds[index] - 2

res.to_csv(RESULT_CSV, index = False)


# In[ ]:


print("Hello world")

