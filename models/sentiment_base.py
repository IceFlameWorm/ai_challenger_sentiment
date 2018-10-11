import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from .base import SingleModel, CompositeModel
from utils.callbacks import CustomModelCheckPoint

class SSingleModel(SingleModel):
    def fit(self, inputs, outputs, *args, **kwargs):
        lr = kwargs['lr']
        EPOCHS = kwargs['epochs']
        BATCH_SIZE = kwargs['batch_size']
        PATIENCE = kwargs['patience']
        FACTOR = kwargs['factor']

        val_x, val_y_onehot = kwargs['validation_data']
        model_file = kwargs['model_file']

        self._model.compile(optimizer = Adam(lr),
                            loss = 'categorical_crossentropy',
                            metrics = ['acc']
                           )

        cbs = [CustomModelCheckPoint(model_file,
                                     monitor = self._f1_monitor,
                                     save_best_only = True,
                                     mode = 'max',
                                     verbose = 1,
                                     val_data=(val_x, val_y_onehot)
                                    ),
               EarlyStopping(patience = PATIENCE),
               ReduceLROnPlateau('val_loss', factor = FACTOR,
                                 verbose = 1,
                                 patience = int(PATIENCE / 2))
              ]

        history = self._model.fit(inputs, outputs,
                                  epochs = EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  validation_data=(val_x, val_y_onehot),
                                  callbacks = cbs,
                                  verbose = 1)

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def _f1_monitor(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = -1)
        y_true = np.argmax(y_true, axis = -1)

        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def predict(self, inputs, *args, **kwargs):
        return np.argmax(self._model.predict(inputs), axis = -1)


class SCompositeModel(CompositeModel):
    def fit(self, inputs, outputs, *args, **kwargs):
        val_x, val_outputs = kwargs['val']
        y_cols = kwargs['y_cols']
        seq = kwargs['seq']
        SAVED_PATH = kwargs['saved_path']

        lr = kwargs.get('lr', 1e-3)
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 64)
        patience = kwargs.get('patience', 6)
        factor = kwargs.get('factor', 0.2)
        model_pre = os.path.join(SAVED_PATH, kwargs.get('model_pre', 'model_'))


        if not os.path.exists(SAVED_PATH):
            os.makedirs(SAVED_PATH)

        train_x = inputs
        for i, comp in enumerate(self._comps):
            col = y_cols[i]
            train_y = outputs[col] + 2
            val_y = val_outputs[col] + 2
            train_y_onehot = to_categorical(train_y)
            val_y_onehot = to_categorical(val_y)
            model_file = model_pre + str(i) + '.h5'
            comp.fit(train_x, train_y_onehot, model_file = model_file,
                     validation_data = (val_x, val_y_onehot),
                     lr = lr, epochs = epochs, batch_size = batch_size,
                     patience = patience, factor = factor)
            comp.load_weights(model_file)
            val_y_pred = comp.predict(val_x)

            F1_score = f1_score(val_y, val_y_pred, average = 'macro')
            print("The %sth grain: f1_score - %s, acc - %s" % (i, F1_score, accuracy_score(val_y, val_y_pred)))
