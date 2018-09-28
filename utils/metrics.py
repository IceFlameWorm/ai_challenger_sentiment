import tensorflow as tf
from keras import backend as K

def micro_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def macro_f1(y_true, y_pred, classes = 4):
    """
    Batch-wise macro-f1
    """
    y_true_labels = K.argmax(y_true, axis = -1)
    pops_list = []
    for i in range(classes):
        pops_list.append(K.cast(K.equal(y_true_labels, i), dtype = tf.float32))

    prps_list = []
    y_pred_labels = K.argmax(y_pred, axis = -1)
    for i in range(classes):
        prps_list.append(K.cast(K.equal(y_pred_labels, i), dtype = tf.float32))

    macro_f1 = 0
    for pops, prps in zip(pops_list, prps_list):
        tps_num = K.sum(K.cast(K.equal(pops + prps, 2), dtype = tf.float32))
        pops_num = K.sum(pops)
        prps_num = K.sum(prps)
        precision = tps_num / (prps_num + K.epsilon())
        recall = tps_num / (pops_num + K.epsilon())
        f1 = 2 * precision * recall /(precision + recall + K.epsilon()) / classes
        macro_f1 += f1

    return macro_f1
