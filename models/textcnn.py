import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from keras.layers import Input, Embedding, Dense, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import SpatialDropout1D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from .base import SingleModel, CompositeModel
from utils.callbacks import CustomModelCheckPoint
from env import *


class SingleTextCNN(SingleModel):
    def _build(self, *args, **kwargs):
        max_len = kwargs['max_len']
        embedding = kwargs['embedding']
        emb_dim = embedding.shape[1]
        inp = Input(shape=(max_len,))
        out1 = self._static_emb(inp, embedding, dim = emb_dim)
        out2 = self._dynamic_emb(inp, embedding, dim = emb_dim)
        out = self._cnn(out1, out2)
        #out = self._dropout(out, rate = 0.5)
        out = self._cls(out)
        model = Model(inputs = inp, outputs = out)
        return model

    def _static_emb(self, inp, embedding, words_num = 50000, dim = 100):
        out = Embedding(words_num,
                        dim, weights = [embedding],
                        trainable=False)(inp)
        return out

    def _dynamic_emb(self, inp, words_num = 50000, dim = 100):
        out = Embedding(words_num, dim,
                        weights = [embedding])(inp)
        return out

    def _cnn(self, inp1, inp2, filter_sizes = [3,4,5], filter_num = 100):
        # Define shared conv layers
        shared_spdropout = SpatialDropout1D(0.5)
        inp1 = shared_spdropout(inp1)
        inp2 = shared_spdropout(inp2)
        shared_convs = []
        for fs in filter_sizes:
            conv = Conv1D(filter_num, fs, activation = 'relu')
            shared_convs.append(conv)

        out1s = []
        out2s = []
        for conv in shared_convs:
            shared_gm = GlobalMaxPooling1D()
            out1 = conv(inp1)
            out1 = shared_gm(out1)
            out1s.append(out1)
            out2 = conv(inp2)
            out2 = shared_gm(out2)
            out2s.append(out2)

        #outs = out1s + out2s
        outs = out2s
        out = Concatenate()(outs)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        return out

    def _dropout(self, inp, rate = 0.5):
        out = Dropout(rate)(inp)
        return out

    def _cls(self, inp):
        out = Dense(4, activation = 'softmax')(inp)
        return out

    def fit(self, inputs, outputs, *args, **kwargs):
        lr = 1e-3
        EPOCHS = 100
        BATCH_SIZE = 64
        PATIENCE = 6

        val_x, val_y_onehot = kwargs['validation_data']
        model_file = kwargs['model_file']

        self._model.compile(optimizer = Adam(lr),
                            loss = 'categorical_crossentropy',
                            metrics = ['acc'])

        cbs = [CustomModelCheckPoint(model_file,
                                     monitor = self._f1_monitor,
                                     save_best_only = True,
                                     mode = 'max',
                                     verbose = 1,
                                     val_data = (val_x, val_y_onehot)),
               EarlyStopping(patience = PATIENCE),
               ReduceLROnPlateau('val_loss', factor=0.2,
                                 verbose = 1,
                                 patience = int(PATIENCE / 2))
              ]

        history = self._model.fit(inputs, outputs, epochs = EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  validation_data=(val_x, val_y_onehot),
                                  callbacks = cbs,
                                  verbose = 1)
        pass

    def _f1_monitor(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = -1)
        y_true = np.argmax(y_true, axis = -1)

        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def predict(self, inputs, *args, **kwargs):
        return np.argmax(self._model.predict(inputs), axis = -1)

class CompositeTextCNN(CompositeModel):
    def fit(self, inputs, outputs, *args, **kwargs):
        val_x, val_outputs = kwargs['val']
        y_cols = kwargs['y_cols']
        seq = kwargs['seq']

        SAVED_PATH = os.path.join(SAVED_MODELS_PATH, 'textcnn')
        if not os.path.exists(SAVED_PATH):
            os.makedirs(SAVED_PATH)

        train_x = inputs
        for i, comp in enumerate(self._comps):
            col = y_cols[i]
            train_y = outputs[col] + 2
            val_y = val_outputs[col] + 2
            train_y_onehot = to_categorical(train_y)
            val_y_onehot = to_categorical(val_y)
            MODEL_PRE = os.path.join(SAVED_PATH, 'textcnn_')
            model_file = MODEL_PRE + str(i) + '.h5'
            comp.fit(train_x, train_y_onehot, model_file = model_file,
                     validation_data = (val_x, val_y_onehot))
            comp.load_weights(model_file)
            val_y_pred = comp.predict(val_x)

            F1_score = f1_score(val_y, val_y_pred, average = 'macro')
            print("The %sth grain: f1_score - %s, acc - %s" % (i, F1_score, accuracy_score(val_y, val_y_pred)))


    pass
