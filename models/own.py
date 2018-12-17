import numpy as np
from .base import SingleModel
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate
from keras.layers import GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from keras.layers import SpatialDropout1D, CuDNNGRU, Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.callbacks import CustomModelCheckPoint
from sklearn.metrics import f1_score

class OwnSingleModel(SingleModel):
    def _build(self, *args, **kwargs):
        max_len = kwargs['max_len']
        embedding = kwargs.get('embedding')
        emb_dim = embedding.shape[1]
        inp = Input(shape = (max_len,))
        out0 = self._base0(inp, embedding=embedding, dim = emb_dim)
        out1 = self._base1(out0)
        out2 = self._base2(out0)
        out3 = self._base3(out0)
        out4 = self._base4(out0)
        out5 = self._base5(out0)
        out6 = self._base6(out0)

        out1_1 = self._cls1_1(out1)
        out1_2 = self._cls1_2(out1)
        out1_3 = self._cls1_3(out1)

        out2_1 = self._cls2_1(out2)
        out2_2 = self._cls2_2(out2)
        out2_3 = self._cls2_3(out2)
        out2_4 = self._cls2_4(out2)

        out3_1 = self._cls3_1(out3)
        out3_2 = self._cls3_2(out3)
        out3_3 = self._cls3_3(out3)

        out4_1 = self._cls4_1(out4)
        out4_2 = self._cls4_2(out4)
        out4_3 = self._cls4_3(out4)
        out4_4 = self._cls4_4(out4)

        out5_1 = self._cls5_1(out5)
        out5_2 = self._cls5_2(out5)
        out5_3 = self._cls5_3(out5)
        out5_4 = self._cls5_4(out5)

        out6_1 = self._cls6_1(out6)
        out6_2 = self._cls6_2(out6)

        inputs = inp
        outputs = [out1_1,
                   out1_2,
                   out1_3,
                   out2_1,
                   out2_2,
                   out2_3,
                   out2_4,
                   out3_1,
                   out3_2,
                   out3_3,
                   out4_1,
                   out4_2,
                   out4_3,
                   out4_4,
                   out5_1,
                   out5_2,
                   out5_3,
                   out5_4,
                   out6_1,
                   out6_2
                  ]
        return Model(inputs = inputs, outputs = outputs)

    def _base0(self, inp, embedding, words_num = 50000, dim = 300):
        """
        Embedding Layers
        """
        if embedding is None:
            out = Embedding(words_num, dim)(inp)
        else:
            out = Embedding(words_num, dim, weights = [embedding], trainable = True)(inp)
        # out = Conv1D(32, 3, activation = 'relu')(out)
        # out = Conv1D(32, 3, activation = 'relu')(out)
        # out = MaxPooling1D(2)(out)
        # out = Conv1D(64, 3, activation = 'relu')(out)
        # out = Conv1D(64, 3, activation = 'relu')(out)
        # out = MaxPooling1D(2)(out)
        # out = Conv1D(128, 3, activation = 'relu')(out)
        # out = Conv1D(128, 3, activation = 'relu')(out)
        # out = MaxPooling1D(2)(out)
        # out = Conv1D(256, 3, activation = 'relu')(out)
        # out = Conv1D(256, 3, activation = 'relu')(out)
        # out = GlobalMaxPooling1D()(out)

        out = SpatialDropout1D(0.5)(out)
        #out = CuDNNGRU(300)(out)
        #out = Bidirectional(CuDNNGRU(200, return_sequences=True))(out)
        #out = Bidirectional(CuDNNGRU(200, return_sequences=True))(out)
        #out1 = GlobalMaxPooling1D()(out)
        #out2 = GlobalAveragePooling1D()(out)
        #out = Concatenate()([out1, out2])
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        return out

    def _basei(self, inp):
        out = inp
        #out = Conv1D(32, 3, activation = 'relu')(inp)
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        #out = MaxPooling1D(2)(out)
        #out = Conv1D(32, 3, activation = 'relu')(out)
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        #out = MaxPooling1D(2)(out)
        #out = GlobalMaxPooling1D()(out)

        # # out = SpatialDropout1D(0.5)(out)
        # out = CuDNNGRU(300, return_sequences = True)(out)
        # #out = SpatialDropout1D(0.5)(out)
        # out = CuDNNGRU(300)(out)
        # #out = Dropout(0.5)(out)
        # #out = BatchNormalization()(out)
        # #out = Dense(200, activation = 'relu')(out)
        # out = Dropout(0.5)(out)
        # out = BatchNormalization()(out)
        out = Bidirectional(CuDNNGRU(300, return_sequences = True))(out)
        out = Bidirectional(CuDNNGRU(300, return_sequences = True))(out)
        y1 = GlobalAveragePooling1D()(out)
        y2 = GlobalMaxPooling1D()(out)
        out = Concatenate()([y1, y2])
        out = Dense(1000, activation = 'relu')(out)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        out = Dense(500, activation= 'relu')(out)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        return out

    def _clsi_j(self, inp):
        out = inp
        # out1 = Conv1D(32, 3, activation= 'relu')(out)
        # out1 = GlobalMaxPooling1D()(out1)
        # out2 = Conv1D(32, 5, activation= 'relu')(out)
        # out2 = GlobalMaxPooling1D()(out2)
        # out3 = Conv1D(32, 7, activation= 'relu' )(out)
        # out3 = GlobalMaxPooling1D()(out3)
        # out = Concatenate(axis= -1)([out1, out2, out3])

        #out = SpatialDropout1D(0.5)(inp)
        #out = CuDNNGRU(100)(out)
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        #out = Dense(100, activation = 'relu')(out)
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        return Dense(4, activation = 'softmax')(out)

    def _base1(self, inp):
        """
        Location
        """
        return self._basei(inp)

    def _cls1_1(self, inp):
        return self._clsi_j(inp)

    def _cls1_2(self, inp):
        return self._clsi_j(inp)

    def _cls1_3(self, inp):
        return self._clsi_j(inp)

    def _base2(self, inp):
        """
        Service
        """
        return self._basei(inp)

    def _cls2_1(self, inp):
        return self._clsi_j(inp)

    def _cls2_2(self, inp):
        return self._clsi_j(inp)

    def _cls2_3(self, inp):
        return self._clsi_j(inp)

    def _cls2_4(self, inp):
        return self._clsi_j(inp)

    def _base3(self, inp):
        """
        Price
        """
        return self._basei(inp)

    def _cls3_1(self, inp):
        return self._clsi_j(inp)

    def _cls3_2(self, inp):
        return self._clsi_j(inp)

    def _cls3_3(self, inp):
        return self._clsi_j(inp)

    def _base4(self, inp):
        """
        Environment
        """
        return self._basei(inp)

    def _cls4_1(self, inp):
        return self._clsi_j(inp)

    def _cls4_2(self, inp):
        return self._clsi_j(inp)

    def _cls4_3(self, inp):
        return self._clsi_j(inp)

    def _cls4_4(self, inp):
        return self._clsi_j(inp)

    def _base5(self, inp):
        """
        Dish
        """
        return self._basei(inp)

    def _cls5_1(self, inp):
        return self._clsi_j(inp)

    def _cls5_2(self, inp):
        return self._clsi_j(inp)

    def _cls5_3(self, inp):
        return self._clsi_j(inp)

    def _cls5_4(self, inp):
        return self._clsi_j(inp)

    def _base6(self, inp):
        """
        Others
        """
        return self._basei(inp)

    def _cls6_1(self, inp):
        return self._clsi_j(inp)

    def _cls6_2(self, inp):
        return self._clsi_j(inp)

    def fit(self, inputs, outputs, *args, **kwargs):
        lr = 1e-3
        BATCH_SIZE = 64
        EPOCHS = 300
        PATIENCE = 6
        model_file = kwargs['model_file']
        val_inputs, val_outputs = kwargs['validation_data']

        self._model.compile(optimizer = Adam(lr),
                            loss = 'categorical_crossentropy',
                            )
        cbs = [
#               ModelCheckpoint(model_file,
#                               monitor = 'val_loss',
#                               save_best_only = True,
#                               mode = 'max'),
               CustomModelCheckPoint(model_file,
                                     monitor = self._f1_monitor,
                                     save_best_only = True,
                                     mode = 'max',
                                     verbose = 1,
                                     val_data = (val_inputs, val_outputs)),
               EarlyStopping(patience = PATIENCE),
               ReduceLROnPlateau('val_loss', factor = 0.2, verbose = 1,
                                 patience = int(PATIENCE / 3))
              ]
        history = self._model.fit(inputs, outputs,
                                  epochs = EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  validation_data = (val_inputs, val_outputs),
                                  callbacks = cbs,
                                  verbose = 1
                                 )
        
    def _f1_monitor(self, y_trues, y_preds):
        y_preds = list(map(lambda x: np.argmax(x, axis = -1), y_preds))
        y_trues = list(map(lambda x: np.argmax(x, axis = -1), y_trues))
        f1s = 0
        for i, (vy, vp) in enumerate(zip(y_trues, y_preds)):
            f1 = f1_score(vy, vp, average='macro')
            f1s += f1
        return f1s / len(y_trues)

    def predict(self, inputs, *args, **kwargs):
        return self._model.predict(inputs)
