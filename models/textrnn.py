from keras.layers import Input, CuDNNGRU, CuDNNLSTM
from keras.layers import GRU
from keras.layers import Embedding, Dense, Concatenate
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
