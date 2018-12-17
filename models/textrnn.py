from keras.layers import Input, CuDNNGRU, CuDNNLSTM
from keras.layers import GRU, Bidirectional, Lambda
from keras.layers import Embedding, Dense, Concatenate
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Conv1D, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import Activation
from keras import backend as K
from keras.models import Model
from .sentiment_base import SSingleModel


class BaseRNN(SSingleModel):
    def _build(self, *args, **kwargs):
        max_len = kwargs['max_len']
        embedding = kwargs['embedding']
        mask_zero = kwargs.get('mask_zero', False)

        inp = Input(shape = (max_len, ))
        out1 = self._static_emb(inp, embedding, mask_zero)
        out2 = self._dynamic_emb(inp, embedding, mask_zero)
        out = self._rnn(out1, out2)
        out = self._cls(out)
        model = Model(inputs = inp, outputs = out)
        return model

    def _static_emb(self, inp, embedding, mask_zero = False, words_num = 50000, dim = 300):
        out = Embedding(words_num,
                        dim, weights = [embedding],
                        mask_zero = mask_zero,
                        trainable=False)(inp)
        return out

    def _dynamic_emb(self, inp, embedding, mask_zero = False, words_num = 50000, dim = 300):
        out = Embedding(words_num, dim,
                        weights = [embedding],
                        mask_zero = mask_zero
                       )(inp)
        return out

    def _rnn(self, inp1, inp2):
        raise NotImplementedError

    def _cls(self, inp):
        out = Dense(4, activation = 'softmax')(inp)
        return out

class CuDNNGRULast(BaseRNN):
    def _rnn(self, inp1, inp2):
        out1, out2 = inp1, inp2
        # shared_spdropout = SpatialDropout1D(0.5)
        # shared_gru = CuDNNGRU(300)
        # out1 = shared_spdropout(out1)
        # out1 = shared_gru(out1)
        # out2 = shared_spdropout(out2)
        # out2 = shared_gru(out2)
        # out = Concatenate()([out1, out2])
        # out = out2
        # out = Dropout(0.5)(out)
        # out = BatchNormalization()(out)

        out = out2
        out = SpatialDropout1D(0.5)(out)
        out = CuDNNGRU(300, return_sequences = True)(out)
        #out = SpatialDropout1D(0.5)(out)(out)
        out = CuDNNGRU(300)(out)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        # out = Dense(300)(out)
        # out = BatchNormalization()(out)
        # out = Activation('relu')(out)
        # out = Dropout(0.5)(out)
        return out

class CuDNNGRUSeq(BaseRNN):
    def _rnn(self, inp1, inp2):
        out1, out2 = inp1, inp2
        # shared_spdropout = SpatialDropout1D(0.5)
        # shared_gru = CuDNNGRU(300)
        # out1 = shared_spdropout(out1)
        # out1 = shared_gru(out1)
        # out2 = shared_spdropout(out2)
        # out2 = shared_gru(out2)
        # out = Concatenate()([out1, out2])
        # out = out2
        # out = Dropout(0.5)(out)
        # out = BatchNormalization()(out)

        out = out2
        out = SpatialDropout1D(0.5)(out)
        out = CuDNNGRU(300, return_sequences = True)(out)
        out = CuDNNGRU(300, return_sequences = True)(out)
        y1 = GlobalAveragePooling1D()(out)
        y2 = GlobalMaxPooling1D()(out)
        out = Concatenate()([y1, y2])
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        # out = Dense(300)(out)
        # out = BatchNormalization()(out)
        # out = Activation('relu')(out)
        # out = Dropout(0.5)(out)
        return out

class GRULast(BaseRNN):
    def _rnn(self, inp1, inp2):
        shared_gru = GRU(100)
        out1 = shared_gru(inp1)
        out2 = shared_gru(inp2)
        out = Concatenate()([out1, out2])
        return out

class BiCuDNNGRULast(BaseRNN):
    def _rnn(self, inp1, inp2):
        out1, out2 = inp1, inp2
        rnn_dim = 300
        #shared_bigru = Bidirectional(CuDNNGRU(rnn_dim, return_sequences=False))
        #shared_shrink = Lambda(lambda x: K.max(x, axis = 1),
        #                       output_shape = (2 *rnn_dim, ))

        #out1 = shared_bigru(inp1)
        #out1 = shared_shrink(out1)

        #out2 = shared_bigru(inp2)
        #out2 = shared_shrink(out2)

        #out = Concatenate()([out1, out2])
        #out = Dropout(0.5)(out)

        out = SpatialDropout1D(0.5)(out2)
        out = Bidirectional(CuDNNGRU(rnn_dim, return_sequences = False))(out)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        return out

class BiCuDNNGRUSeq(BaseRNN):
    def _rnn(self, inp1, inp2):
        out1, out2 = inp1, inp2
        rnn_dim = 300
        #shared_spadropout = SpatialDropout1D(0.5)
        #shared_bigru = Bidirectional(CuDNNGRU(rnn_dim, return_sequences=True))
        #shared_shrink = Lambda(lambda x: K.mean(x, axis = 1),
        #                       output_shape = (2 *rnn_dim, ))
        #shared_gm = GlobalMaxPooling1D()
        #shared_ga = GlobalAveragePooling1D()

        #out1 = shared_bigru(inp1)
        #out1 = shared_shrink(out1)
        #out1 = shared_gm(out1)

        #out2 = shared_bigru(inp2)
        #out2 = shared_shrink(out2)
        #out2 = shared_gm(out2)

        #out = Concatenate()([out1, out2])
        #out = out2
        #out = Dropout(0.5)(out)
        #out = BatchNormalization()(out)
        out = SpatialDropout1D(0.5)(out2)
        out = Bidirectional(CuDNNGRU(rnn_dim, return_sequences = True))(out)
        out = GlobalMaxPooling1D()(out)
        out = Dropout(0.5)(out)
        out = BatchNormalization()(out)
        return out

class SimpleRCNN(BaseRNN):
    def _rnn(self, inp1, inp2):
        rnn_dim = 100
        shared_bigru = Bidirectional(CuDNNGRU(rnn_dim, return_sequences=True))
        # shared_shrink = Lambda(lambda x: K.mean(x, axis = 1),
        #                        output_shape = (2 *rnn_dim, ))

        out1 = shared_bigru(inp1)
        #out1 = shared_shrink(out1)

        out2 = shared_bigru(inp2)
        #out2 = shared_shrink(out2)

        cnn_dim = 100
        shared_conv = Conv1D(cnn_dim, 1, activation = 'tanh')
        out3 = shared_conv(inp1)
        out4 = shared_conv(inp2)
        out = Concatenate()([out1,
                             out2,
                             out3,
                             out4
                            ])
        out = GlobalMaxPooling1D()(out)
        out = BatchNormalization()(out)
        out = Dropout(0.5)(out)
        return out
