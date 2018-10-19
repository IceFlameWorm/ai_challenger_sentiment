import math
import numpy as np
from keras.utils import to_categorical
from collections import Counter

def oversample(xs, ys, batch_size, classes = 4,
               random_state = 7777
              ):
    """
    Args:
        xs: feature vectors
        ys: targets
    """
    assert batch_size >= 4, "batch size must be greater than classes, classes is %d" % classes
    sample_num = len(ys)
    loop_count = int(math.ceil(sample_num / batch_size))

    ys_onehot = to_categorical(ys)
    xs_pc = []
    ys_onehot_pc = []
    for c in range(classes):
        xs_pc.append(xs[ys == c])
        ys_onehot_pc.append(ys_onehot[ys == c])

    size_pc = Counter([x % classes for x in range(batch_size)])

    np.random.seed(random_state)
    while True:
        xs_list, ys_list = [], []
        for c, size in size_pc.items():
            c_xs, c_ys_onehot = xs_pc[c], ys_onehot_pc[c]
            c_num = len(c_ys_onehot)
            indices = np.random.choice(c_num, size)
            rnd_xs = c_xs[indices]
            rnd_ys = c_ys_onehot[indices]
            xs_list.append(rnd_xs)
            ys_list.append(rnd_ys)

        batch_xs = np.vstack(xs_list)
        batch_ys = np.vstack(ys_list)
        yield batch_xs, batch_ys
    pass
