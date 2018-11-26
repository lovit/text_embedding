import scipy as sp
from collections import defaultdict

from .utils import get_process_memory


def scan_vocabulary(sents, min_count):
    """
    :param sents: list of list of str (like)
        utils.Word2VecCorpus
    :param min_count: int
        Minimum number of word frequency

    It returns
    ----------
    vocab_to_idx: dict {str:int}
        vocabulary to index mapper
    idx_to_vocab: list of str
        vocabluary list
    """

    counter = defaultdict(int)
    for sent in sents:
        for word in sent:
            counter[word] += 1
    counter = {word:count for word, count in counter.items()
               if count >= min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter,
                    key=lambda x:-counter[x])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return vocab_to_idx, idx_to_vocab

def dict_to_sparse(dd, row_to_idx, col_to_idx, n_rows=-1, n_cols=-1):
    """
    :param dd: nested dict
        dict[str][str] = float
    :param row_to_idx: dict {str:int}
        Row value encoder
    :param col_to_idx: dict {str:int}
        Column value encoder
    :param n_rows: int
        Size of rows. default is len(row_to_idx)
    :param n_cols: int
        Size of rows. default is len(col_to_idx)

    It returns
    ----------
    X: scipy.sparse.csr.csr_matrix
        (n_rows, n_cols) shape sparse matrix
    """

    if n_rows == -1:
        n_rows = len(row_to_idx)
    if n_cols == -1:
        n_cols = len(col_to_idx)

    rows, cols, data = [], [], []
    for r, col_val in dd.items():
        i = row_to_idx[r]
        for c, val in col_val.items():
            j = col_to_idx[c]
            rows.append(i)
            cols.append(j)
            data.append(val)
    X = sp.sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return X

def word_context(sents, vocab_to_idx, windows=3, min_count=1,
    dynamic_weight=True, verbose=True, row_vocabs=None, prune_point=500000):
    """
    :param sents: list of list of str (like)
        utils.Word2VecCorpus
    :param vocab_to_idx: dict {str:int}
        vocabulary to index mapper
    :param windows: int
        Size of context range. default is 3
    :param min_count: int
        Minimum number of co-occurrence count
    :param dynamic_weight: Boolean
        If True, the weight is [1, ..., 1/window] such as [1, 2/3, 1/3]
        Default is True
    :param verbose: Boolean
        If True, show progress
        Deefault is True
    :param row_vocabs: dict or set
        Row words. Default is all words in vocab_to_idx
    :param prune_point: int
        Number of sents to prune with min_count

    It returns
    ----------
    dd: nested dict
        dict[word][context] = frequency
    """

    if not row_vocabs:
        row_vocabs = {vocab for vocab in vocab_to_idx}

    w = windows
    if dynamic_weight:
        weight = [(w - i) / w for i in range(w)]
    else:
        weight = [1] * windows

    dd = defaultdict(lambda: defaultdict(int))
    for i_sent, words in enumerate(sents):
        if min_count > 1 and i_sent % prune_point == 0:
            dd = _prune(dd, min_count)
        if verbose and i_sent % 1000 == 0:
            print('\r(word, context) from {} sents ({:.3f} Gb)'.format(
                i_sent, get_process_memory()), end='')
        if not words:
            continue

        n = len(words)
        for i, word in enumerate(words):
            if not (word in row_vocabs):
                continue
            # left_contexts
            for w in range(windows):
                j = i - (w + 1)
                if j < 0 or not (words[j] in vocab_to_idx):
                    continue
                dd[word][words[j]] += weight[w]
            # right_contexts
            for w in range(windows):
                j = i + w + 1
                if j >= n or not (words[j] in vocab_to_idx):
                    continue
                dd[word][words[j]] += weight[w]

    if verbose:
        print('\r(word, context) was constructed from {} sents ({} words, {:.3f} Gb)'.format(
            i_sent, len(vocab_to_idx), get_process_memory()))
    return dd

def _prune(dd, min_count):
    dd_ = defaultdict(lambda: defaultdict(int))
    for k1, d in dd.items():
        d_ = defaultdict(int, {k2:v for k2, v in d.items() if v >= min_count})
        if len(d_) > 0:
            dd_[k1] = d_
    return dd_

def label_word(labeled_corpus, vocab_to_idx, verbose=True):
    """
    :param labeled_corpus: utils.Doc2VecCorpus (like)
        It yield (labels, words)
        labels and words are list of str
    :param vocab_to_idx: dict {str:int}
        vocabulary to index mapper
    :param verbose: Boolean
        If True, show progress
        Deefault is True

    It returns
    ----------
    label_to_idx: dict {str:int}
        label to index mapper
    dd: nested dict
        dict[label][word] = frequency
    """

    if verbose:
        print('(label, word) construction ...', end='')

    label_to_idx = defaultdict(lambda: len(label_to_idx))
    dd = defaultdict(lambda: defaultdict(int))
    for i, (labels, words) in enumerate(labeled_corpus):
        words = [word for word in words if word in vocab_to_idx]
        if not words:
            continue
        for label in labels:
            _ = label_to_idx[label]
            for word in words:
                dd[label][word] += 1

    if verbose:
        args = (len(label_to_idx), len(vocab_to_idx))
        print('\r(label, word) was constructed (%d labels, %d words)' % args)

    return label_to_idx, dd