import os
import numpy as np
from sklearn.utils import class_weight
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Embedding, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dropout, BatchNormalization, SpatialDropout1D
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger
from env import *

from utils.metrics import macro_f1
from utils.clr_callback import CyclicLR

from .base import SingleModel, CompositeModel


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
    #out = SpatialDropout1D(0.5)(out)
    out = Conv1D(64, 3, activation='relu')(out)
    out = MaxPooling1D(5)(out)
    #out = BatchNormalization()(out)
    #out = Dropout(0.5)(out)
    out = Conv1D(64, 3, activation='relu')(out)
    #out = Dropout(0.5)(out)
    out = GlobalMaxPooling1D()(out)
    #out = BatchNormalization()(out)
    out = Dense(4, activation='softmax')(out)
    model = Model(inputs = inp, outputs = out)
    return model


def train_CNN(train, val, test, y_cols, debug = False, seq = 'word_seq'):
    F1_scores = 0
    F1_score = 0

    if debug:
        y_cols = ['location_traffic_convenience']

    train_x = np.array(train[seq].values.tolist(), dtype = np.int)
    val_x = np.array(val[seq].values.tolist(), dtype = np.int)
    test_x = np.array(test[seq].values.tolist(), dtype = np.int)

    model = build_model(maxlen = train_x.shape[1])
    lr = 1e-5
    model.compile(optimizer= Adam(lr),loss='categorical_crossentropy',metrics=['acc',
                                                                                 #macro_f1
                                                                                ])

    EPOCHS = 200
    BATCH_SIZE = 64
    PATIENCE = 6
    MODEL_PRE = os.path.join(SAVED_MODELS_PATH, 'linxi', 'cnn_')

    F1_score = 0
    F1_scores = 0
    res = test.drop(seq, axis = 1)
    for index, col in enumerate(y_cols):
        MODEL_FILE = MODEL_PRE + str(index) + '.h5'
        cbs = [ModelCheckpoint(MODEL_FILE, monitor= 'val_acc', save_best_only=True, mode = 'max'),
               EarlyStopping(patience=PATIENCE),
               #ReduceLROnPlateau('val_loss', factor=0.2, verbose=1, patience=int(PATIENCE / 2)),
               CyclicLR(base_lr = lr, max_lr = 10 * lr, step_size = 1000,
                        #mode = 'exp_range', gamma=0.99994
                       )
              ]
        train_y = train[col] + 2
        val_y = val[col] + 2

        train_sample_weights = class_weight.compute_class_weight('balanced', train_y, train_y)
        val_sample_weights = class_weight.compute_class_weight('balanced', val_y, train_y)

        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)

        history = model.fit(train_x, y_train_onehot, epochs = EPOCHS,
                            batch_size = BATCH_SIZE, validation_data = (val_x, y_val_onehot,
                                                                        #val_sample_weights
                                                                       ),
                            callbacks = cbs,
                            #sample_weight = train_sample_weights,
                            verbose = 1)

        model = load_model(MODEL_FILE,
                           #custom_objects = {"macro_f1": macro_f1}
                          )
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


class LinXiSingleModel(SingleModel):
    def _build(self, *args, **kwargs):
        maxlen = kwargs['maxlen']

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
        #out = SpatialDropout1D(0.5)(out)
        out = Conv1D(64, 3, activation='relu')(out)
        out = MaxPooling1D(5)(out)
        #out = BatchNormalization()(out)
        #out = Dropout(0.5)(out)
        out = Conv1D(64, 3, activation='relu')(out)
        #out = Dropout(0.5)(out)
        out = GlobalMaxPooling1D()(out)
        #out = BatchNormalization()(out)
        out = Dense(4, activation='softmax')(out)
        model = Model(inputs = inp, outputs = out)
        return model


    def fit(self, inputs, outputs, *args, **kwargs):
        lr = 1e-3
        EPOCHS = 1
        BATCH_SIZE = 64
        PATIENCE = 6
        model_file = kwargs['model_file']
        self._model.compile(optimizer= Adam(lr),loss='categorical_crossentropy',metrics=['acc'])
        cbs = [ModelCheckpoint(model_file, monitor= 'val_acc', save_best_only=True, mode = 'max'),
               # EarlyStopping(patience=PATIENCE),
               # ReduceLROnPlateau('val_loss', factor=0.2, verbose=1, patience=int(PATIENCE / 2)),
               #CyclicLR(base_lr = lr, max_lr = 10 * lr, step_size = 1000,
                        #mode = 'exp_range', gamma=0.99994
               #        )
              ]
        train_x = inputs
        y_train_onehot = outputs
        val_x, y_val_onehot = kwargs['validation_data']
        history = self._model.fit(train_x, y_train_onehot, epochs = EPOCHS,
                            batch_size = BATCH_SIZE, validation_data = (val_x, y_val_onehot,
                                                                        #val_sample_weights
                                                                       ),
                            callbacks = cbs,
                            #sample_weight = train_sample_weights,
                            verbose = 1)

    def _load(self, model_file, weights_file = None):
        return load_model(model_file)

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def predict(self, inputs, *args, **kwargs):
        return np.argmax(self._model.predict(inputs), axis = -1)


class LinXiCompositeModel(CompositeModel):
    def fit(self, inputs, outputs, *args, **kwargs):
        val_x, val_outputs = kwargs['val']
        y_cols = kwargs['y_cols']
        seq = kwargs['seq']

        train_x = inputs

        for i, comp in enumerate(self._comps):
            col = y_cols[i]
            train_y = outputs[col] + 2
            val_y = val_outputs[col] + 2
            y_train_onehot = to_categorical(train_y)
            y_val_onehot = to_categorical(val_y)
            MODEL_PRE = os.path.join(SAVED_MODELS_PATH, 'linxi', 'cnn_')
            model_file = MODEL_PRE + str(i) + '.h5'
            comp.fit(train_x, y_train_onehot, model_file = model_file, validation_data = (val_x, y_val_onehot))
            comp._load(model_file)

            y_val_pred = comp.predict(val_x)

            F1_score = f1_score(val_y, y_val_pred, average = 'macro')
            print("The %sth grain: f1_score - %s, acc - %s" % (i, F1_score, accuracy_score(val_y, y_val_pred)))

    def predict(self, inputs, *args, **kwargs):
        seq = kwargs['seq']
        xs = np.array(inputs[seq].values.tolist(), dtype = np.int)

        outs = super(LinXiCompositeModel, self).predict(xs)
        return outs


if __name__ == "__main__":
    pass
