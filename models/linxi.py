import os
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Embedding, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger
from env import *


def build_model(maxlen):
   # model = Sequential()
   # embedding_dim = 128
   # model.add(Embedding(50000, embedding_dim,input_length=maxlen))
   # model.add(Conv1D(64, 3, activation='relu'))
   # model.add(MaxPooling1D(5))
   # # model.add(Dropout(0.5))
   # model.add(Conv1D(64, 3, activation='relu'))
   # # model.add(Dropout(0.5))
   # model.add(GlobalMaxPooling1D())
   # # model.add(layers.Dense(32, activation='relu'))
   # model.add(Dense(4, activation='softmax'))
    inp = Input(shape=(maxlen,))
    out = Embedding(50000, 128)(inp)
    out = Conv1D(64, 3, activation='relu')(out)
    out = MaxPooling1D(5)(out)
    out = Conv1D(64, 3, activation='relu')(out)
    out = GlobalMaxPooling1D()(out)
    out = Dense(4, activation='softmax')(out)
    model = Model(inputs = inp, outputs = out)
    return model


def train_CNN(train, val, test, y_cols, debug = False):
    F1_scores = 0
    F1_score = 0

    if debug:
        y_cols = ['location_traffic_convenience']

    train_x = np.array(train['word_seq'].values.tolist(), dtype = np.int)
    val_x = np.array(val['word_seq'].values.tolist(), dtype = np.int)
    test_x = np.array(test['word_seq'].values.tolist(), dtype = np.int)

    model = build_model(maxlen = train_x.shape[1])
    model.compile(optimizer= Adam(1e-4),loss='categorical_crossentropy',metrics=['acc'])

    EPOCHS = 30
    BATCH_SIZE = 64
    PATIENCE = 6
    MODEL_PRE = os.path.join(SAVED_MODELS_PATH, 'linxi', 'cnn_')

    F1_score = 0
    F1_scores = 0
    res = test.drop('word_seq', axis = 1)
    for index, col in enumerate(y_cols):
        MODEL_FILE = MODEL_PRE + str(index)
        cbs = [ModelCheckpoint(MODEL_FILE, monitor='val_acc',save_best_only=True),
               EarlyStopping(patience=PATIENCE),
               ReduceLROnPlateau('val_loss', factor=0.2, verbose=1, patience=int(PATIENCE / 2))]
        train_y = train[col] + 2
        val_y = val[col] + 2

        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)

        history = model.fit(train_x, y_train_onehot, epochs = EPOCHS,
                            batch_size = BATCH_SIZE, validation_data = (val_x, y_val_onehot),
                            callbacks = cbs,
                            verbose = 1)

        model = load_model(MODEL_FILE)
        y_test_prob = model.predict(test_x) # scores
        y_val_prob = model.predict(val_x)

        y_val_pred = np.argmax(y_val_prob, axis = 1)
        y_test_pred = np.argmax(y_test_prob, axis = 1)

        F1_score = f1_score(val_y, y_val_pred, average = 'macro')
        F1_scores += F1_score
        print("The %sth grain: f1_score - %s, acc - %s" % (index, F1_score, accuracy_score(val_y, y_val_pred)))
        res[col] = y_test_pred - 2

    print('All f1 score: %s' % (F1_scores / len(y_cols)))
    return res


def main():
    pass


if __name__ == "__main__":
    pass
