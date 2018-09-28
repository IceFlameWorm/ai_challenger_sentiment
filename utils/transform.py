import jieba
from tqdm import tqdm
from snownlp import SnowNLP
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def gen_stopwords(fn):
    """
    Create a stopwords list from a stopwords file
    """
    with open(fn, encoding='utf8') as f:
        return [word.strip() for word in f]


def segment_words(series, stopwords = None):
    """
    Segment words and remove the first and last words(quotation marks)
    """
    pbar = tqdm(total = len(series))
    def seg(x):
        words = list(jieba.cut(x))[1:-1]
        if stopwords is None:
            res = words
        else:
            res = [word for word in words if word not in stopwords]
        pbar.update(1)
        return res

    res =  series.apply(seg)
    pbar.close()
    return res


def rm_stopwords(series, stopwords):

    def rm(item):
        words = list(item)
        return [word for word in words if word not in stopwords]

    return series.apply(rm)


def text2seqs(train_texts, val_texts, test_texts):
    """
    Args:
        train_texts: list of words
        val_texts: list of words
        test_texts: list of words

    Returns:
        train_seqs: list of integers
        val_seqs: list of integers
        test_seqs: list of integers
    """
    tokenizer = Tokenizer(num_words = 50000)
    all_texts = train_texts + val_texts + test_texts
    tokenizer.fit_on_texts(all_texts)
    train_seqs = tokenizer.texts_to_sequences(train_texts)
    val_seqs = tokenizer.texts_to_sequences(val_texts)
    test_seqs = tokenizer.texts_to_sequences(test_texts)
    return (train_seqs, val_seqs, test_seqs), tokenizer.word_index


def padseqs(train_seqs, val_seqs, test_seqs):
    """
    Args:
        list of integers
    """
    all_seqs = train_seqs + val_seqs + test_seqs
    max_len = max(map(len, all_seqs))
    train_seqs_padded = pad_sequences(train_seqs, maxlen = max_len)
    val_seqs_padded = pad_sequences(val_seqs, maxlen = max_len)
    test_seqs_padded = pad_sequences(test_seqs, maxlen = max_len)
    return (train_seqs_padded, val_seqs_padded, test_seqs_padded), max_len


def trad2simple(series):
    pbar = tqdm(total = len(series))

    def _tran2simple(x):
        s = SnowNLP(x)
        pbar.update(1)
        return s.han

    res = series.apply(_tran2simple)
    pbar.close()
    return res
