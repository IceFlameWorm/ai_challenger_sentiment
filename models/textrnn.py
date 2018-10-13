from keras.layers import Input, CuDNNGRU, CuDNNLSTM
from keras.layers import GRU, Bidirectional, Lambda
from keras.layers import Embedding, Dense, Concatenate
from keras.layers import Dropout
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
        out2 = self._dynamic_emb(inp, mask_zero)
        out = self._rnn(out1, out2)
        out = self._cls(out)
        model = Model(inputs = inp, outputs = out)
        return model

    def _static_emb(self, inp, embedding, mask_zero = False, words_num = 50000, dim = 100):
        out = Embedding(words_num,
                        dim, weights = [embedding],
                        mask_zero = mask_zero,
                        trainable=False)(inp)
        return out

    def _dynamic_emb(self, inp, mask_zero = False, words_num = 50000, dim = 100):
        out = Embedding(words_num, dim,
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
        shared_gru = CuDNNGRU(100)
        out1 = shared_gru(inp1)
        out2 = shared_gru(inp2)
        out = Concatenate()([out1, out2])
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
        rnn_dim = 100
        shared_bigru = Bidirectional(CuDNNGRU(rnn_dim, return_sequences=False))
        #shared_shrink = Lambda(lambda x: K.max(x, axis = 1),
        #                       output_shape = (2 *rnn_dim, ))

        out1 = shared_bigru(inp1)
        #out1 = shared_shrink(out1)

        out2 = shared_bigru(inp2)
        #out2 = shared_shrink(out2)

        out = Concatenate()([out1, out2])
        #out = Dropout(0.5)(out)
        return out

class BiCuDNNGRUSeq(BaseRNN):
    def _rnn(self, inp1, inp2):
        rnn_dim = 100
        shared_bigru = Bidirectional(CuDNNGRU(rnn_dim, return_sequences=True))
        shared_shrink = Lambda(lambda x: K.mean(x, axis = 1),
                               output_shape = (2 *rnn_dim, ))

        out1 = shared_bigru(inp1)
        out1 = shared_shrink(out1)

        out2 = shared_bigru(inp2)
        out2 = shared_shrink(out2)

        out = Concatenate()([out1, out2])
        out = Dropout(0.5)(out)
        return out
