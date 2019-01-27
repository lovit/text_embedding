import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot
from .keyword import proportion_keywords
from .math import train_pmi
from .vectorizer import dict_to_sparse
from .vectorizer import label_word
from .vectorizer import scan_vocabulary
from .word2vec import Word2Vec

class Doc2Vec(Word2Vec):
    """
    :param sentences: list of list of str (like)
        Iterable of iterables, optional
        A sentence is represented with list of str.
    :param size: int. passed to :py:func:`sklearn.decomposition.TruncatedSVD`.
        Word vector dimension
        Default is 100
    :param window: int
        The number of context words is 2 x window
        Default is 3
    :param min_count: int
        Minumum frequency of words
        Default is 10
    :param negative: int. passed to :py:func:`.pmi`.
        Number of negative samples. Minimum PMI is automatically
        defined with this value; log(negative)
        Default is 10
    :param alpha: float. passed to :py:func:`.pmi`.
        Nonnegative, PMI smoothing factor
        Default is 0.0
    :param beta: float. passed to :py:func:`.pmi`.
        0 < beta <= 1, PMI smoothing factor.
        PMI_xy = log( Pxy / (Px x Py^beta) )
        Default is 0.75
    :param dynamic_weight: Boolean. passed to :py:func:`.vectorizer`.
        Use dynamic weight such as [1/3, 2/3, 3/3] for windows = 3 if True
    :param verbose: Boolean
        Verbose mode if True
    :param n_iter: int
        Number of SVD iteration.
        Default is 5
    :param min_cooccurrence: int
        Minimum number of co-occurrence count
    :param prune_point: int
        Number of sents to prune with min_count
    """

    def __init__(self, sentences=None, size=100, window=3, min_count=10,
        negative=10, alpha=0.0, beta=0.75, dynamic_weight=False,
        verbose=True, n_iter=5, min_cooccurrence=5, prune_point=500000):

        super().__init__(sentences, size, window,
            min_count, negative, alpha, beta, dynamic_weight,
            verbose, n_iter, min_cooccurrence, prune_point)

    def train(self, doc2vec_corpus):
        """
        :param doc2vec_corpus: utils.Doc2VecCorpus (like)
            It yield (labels, sent).
            The form of sent and labels are list of str
        """

        if self.is_trained:
            raise ValueError('Doc2Vec model already trained')

        if not hasattr(doc2vec_corpus, 'yield_label'):
            raise ValueError('Input argument format is incorrect')

        doc2vec_corpus.yield_label = False
        self._vocab_to_idx, self._idx_to_vocab, self._idx_to_count = scan_vocabulary(
            doc2vec_corpus, min_count=self._min_count, verbose=self._verbose)
        self._vocab_to_idx_ = dict(self._vocab_to_idx.items())

        WW = self._make_word_context_matrix(
            doc2vec_corpus, self._vocab_to_idx)

        doc2vec_corpus.yield_label = True
        DW, self._label_to_idx = self._make_label_word_matrix(
            doc2vec_corpus, self._vocab_to_idx)
        self._idx_to_label = [label for label, idx
            in sorted(self._label_to_idx.items(), key=lambda x:x[1])]

        X = self._make_stacked_matrix(WW, DW)

        pmi, px, py = train_pmi(X, beta=self._beta, min_pmi=0)

        n_vocab = WW.shape[0]
        n_label = DW.shape[0]
        py_vocab = px[:,:n_vocab]
        py_vocab /= py_vocab.sum()
        self._py = py_vocab

        if self._verbose:
            print('train SVD ... ', end='')

        representation, transformer = self._get_repr_and_trans(pmi)

        self.wv = representation[:n_vocab]
        #self.dv = representation[n_vocab:]
        self._transformer = transformer[:n_vocab]
        self.n_vocabs = n_vocab
        self.dv = self.infer_docvec_from_vector(DW)

        if self._verbose:
            print('done')

        self._transformer_ = self._get_word2vec_transformer(WW)

    def _make_label_word_matrix(self, doc2vec_corpus, vocab_to_idx):
        label_to_idx, DWd = label_word(doc2vec_corpus, vocab_to_idx)
        DW = dict_to_sparse(
            dd = DWd,
            row_to_idx = label_to_idx,
            col_to_idx = vocab_to_idx)

        return DW, label_to_idx

    def _make_stacked_matrix(self, WW, DW):
        n_vocab = WW.shape[0]
        n_label = DW.shape[0]
        WD_W = sp.sparse.vstack([WW, DW])

        WD = DW.copy().transpose()
        rows, cols = WD.nonzero()
        data = WD.data
        WD_D = sp.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_vocab + n_label, n_label))
        X = sp.sparse.hstack([WD_W, WD_D]).tocsr()
        return X

    def _get_word2vec_transformer(self, WW):
        pmi_ww, _, _ = train_pmi(WW, py=self._py, beta=self._beta, min_pmi=0)
        _, transformer = self._get_repr_and_trans(pmi_ww)
        return transformer

    def _get_label_influence(self, pmi_ww):
        diff = compute_embedding_difference(
            self._transformer_, self._transformer, pmi_ww)
        influence = diff.mean()
        return influence, diff

    def similar_docs_from_bow(self, bow, topk=10):
        pmi_dw, _, _ = train_pmi(bow, py=self._py, beta=1, min_pmi=0)
        y = safe_sparse_dot(pmi_dw, self._transformer)
        return self.similar_docs_from_vector(y, topk)

    def similar_docs_from_vector(self, vector, topk=10):
        dist = pairwise_distances(vector, self.dv, metric='cosine')[0]
        similars = []
        for similar_idx in dist.argsort():
            if len(similars) >= topk:
                break
            similar_word = self._idx_to_label[similar_idx]
            similars.append((similar_word, 1-dist[similar_idx]))
        return similars

    def infer_docvec_from_corpus(self, doc2vec_corpus):
        DW, label_to_idx = self._make_label_word_matrix(
            doc2vec_corpus, self._vocab_to_idx)
        return self.infer_docvec_from_vector(DW, label_to_idx)

    def infer_docvec_from_vector(self, bow, label_to_idx=None):
        y = self.infer_wordvec_from_vector(
            bow, row_to_vocab=None, append=False)
        if label_to_idx is None:
            return y
        else:
            idx_to_label = [label for label in
                sorted(label_to_idx, key=lambda x:label_to_idx[x])]
            return y, idx_to_label

def label_proportion_keywords(doc2vec_model, doc2vec_corpus, is_stopword=None):

    doc2vec_corpus.yield_label = True
    vocab_to_idx = doc2vec_model._vocab_to_idx
    idx_to_vocab = doc2vec_model._idx_to_vocab

    # get label - term matrix
    DW, label_to_idx = doc2vec_model._make_label_word_matrix(
        doc2vec_corpus, vocab_to_idx)
    idx_to_label = [label for label in sorted(
        label_to_idx, key=lambda x:label_to_idx[x])]

    # train pmi
    pmi, _, _ = train_pmi(DW, doc2vec_model._py, min_pmi=0)

    # frequency weighted pmi
    n_labels, n_terms = DW.shape
    DW = DW.toarray()
    pmi_ = np.zeros((n_labels, n_terms))
    for i in range(n_labels):
        pmi_[i] = safe_sparse_dot(pmi[i].reshape(-1), np.diag(DW[i]))

    # extract keywords
    keywords = proportion_keywords(pmi_,
        index2word=idx_to_vocab, is_stopword=is_stopword)
    keywords = [(label, keyword) for keyword, label
        in zip(keywords, idx_to_label)]

    return keywords

def label_influence(doc2vec_model, doc2vec_corpus,
    batch_size=1000, topk=100, verbose=True):

    doc2vec_corpus.yield_label = False
    WW = doc2vec_model._make_word_context_matrix(
        doc2vec_corpus, doc2vec_model._vocab_to_idx)
    pmi_ww, _, _ = train_pmi(WW, py=doc2vec_model._py, beta=1, min_pmi=0)

    n = pmi_ww.shape[0]
    wvw = safe_sparse_dot(pmi_ww, w2v_transformer)
    wvd = safe_sparse_dot(pmi_ww, d2v_transformer)

    diff = np.zeros(n)
    max_batch = math.ceil(n / batch_size)
    for batch in range(max_batch):
        b = batch * batch_size
        e = min((batch + 1) * batch_size, n)
        dist_w = pairwise_distances(wvw[b:e], wvw, metric='cosine')
        dist_d = pairwise_distances(wvd[b:e], wvd, metric='cosine')
        dist = abs(dist_w - dist_d)
        dist.sort(axis=1)
        dist = dist[:,-topk:].mean(axis=1)
        diff[b:e] = dist

        if verbose:
            print('\rcomputing label influence %d / %d' % (batch+1, max_batch), end='')
    if verbose:
        print('\rcomputing label influence %d / %d done' % (max_batch, max_batch))

    return diff
