from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dense
from keras.layers import GlobalAveragePooling1D
from .sentiment_base import SSingleModel

class FastText(SSingleModel):
    def _build(self, *args, **kwargs):
        max_len = kwargs['max_len']
        embedding = kwargs['embedding']
        mask_zero = kwargs.get('mask_zero', False)

        inp = Input(shape = (max_len, ))
        out = self._dynamic_emb(inp, mask_zero)
        out = self._fasttext(out)
        out = self._cls(out)
        model = Model(inputs = inp, outputs = out)
        return model

    def _dynamic_emb(self, inp, mask_zero = False, words_num = 50000, dim = 100):
        out = Embedding(words_num, dim,
                        mask_zero = mask_zero
                       )(inp)
        return out

    def _fasttext(self, inp):
        out = GlobalAveragePooling1D()(inp)
        return out

    def _cls(self, inp):
        out = Dense(4, activation = 'softmax')(inp)
        return out
