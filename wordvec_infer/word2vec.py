import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot
from .math import train_pmi
from .math import fit_svd
from .vectorizer import dict_to_sparse
from .vectorizer import scan_vocabulary
from .vectorizer import word_context


class Word2Vec:

    def __init__(self, sentences=None, size=100, window=3, min_count=10,
        negative=10, alpha=0.0, beta=0.75, dynamic_weight=False,
        verbose=True, n_iter=5):

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
        """

        # user defined parameters
        self._size = size
        self._window = window
        self._min_count = min_count
        self._negative = negative
        self._alpha = alpha
        self._beta = beta
        self._dynamic_weight = dynamic_weight
        self._verbose = verbose
        self._n_iter = n_iter

        # trained attributes
        self.wv = None # word vector
        self._vocab_to_idx = None
        self._vocab_to_idx_ = None # include appended words
        self._idx_to_vocab = None
        self._py = None
        self._transformer = None
        self.n_vocabs = 0

        if sentences:
            self.train(sentences)

    @property
    def is_trained(self):
        return self.wv is not None

    def train(self, word2vec_corpus):
        """
        :param word2vec_corpus: utils.Word2VecCorpus (like)
            It yield sent. The form of sent is list of str
        """

        if self.is_trained:
            raise ValueError('Word2Vec model already trained')

        self._vocab_to_idx, self._idx_to_vocab = scan_vocabulary(
            word2vec_corpus, min_count=self._min_count)
        self._vocab_to_idx_ = dict(self._vocab_to_idx.items())

        WWd = word_context(
            sents = word2vec_corpus,
            windows = self._window,
            dynamic_weight = self._dynamic_weight,
            verbose = self._verbose,
            vocab_to_idx = self._vocab_to_idx)

        WW = dict_to_sparse(
            dd = WWd,
            row_to_idx = self._vocab_to_idx,
            col_to_idx = self._vocab_to_idx)

        pmi_ww, px, self._py = train_pmi(
            WW, beta = self._beta, min_pmi = 0)

        if self._verbose:
            print('train SVD ... ', end='')

        U, S, VT = fit_svd(pmi_ww, n_components=self._size, n_iter=self._n_iter)
        S_ = S ** (0.5)
        self.wv = U * S_
        self._transformer = VT.T * (S_ ** (-1))
        self.n_vocabs = self.wv.shape[0]

        if self._verbose:
            print('done')

    def similar_words(self, word, topk=10):
        """
        :param word: str
            Query word
        :param topk: int
            Number of most similar words. default is 10

        It returns
        ----------
        similars : list of tuple
            List of tuple of most similar words.
            Each tuple consists with (word, cosine similarity)
        """

        query_idx = self._vocab_to_idx_.get(word, -1)

        if query_idx < 0:
            return []

        query_vector = self.wv[query_idx,:].reshape(1,-1)
        return self.similar_words_from_vector(query_vector, topk, query_idx)

    def similar_words_from_vector(self, vector, topk=10, query_idx=-1):
        """
        :param vector: numpy.ndarray
            A vector of query word. Its shape should be (1, self._size)
        :param topk: int
            Number of most similar words
        :param query_idx: int
            Word idx that to be excluded in similarity search result.

        It returns
        ----------
        similars : list of tuple
            List of tuple of most similar words.
            Each tuple consists with (word, cosine similarity)
        """

        assert vector.shape == (1, self._size)

        dist = pairwise_distances(vector, self.wv, metric='cosine')[0]

        similars = []
        for similar_idx in dist.argsort():
            if similar_idx == query_idx:
                continue
            if len(similars) >= topk:
                break
            similar_word = self._idx_to_vocab[similar_idx]
            similars.append((similar_word, 1-dist[similar_idx]))

        return similars

    def infer_wordvec(self, word2vec_corpus, word_set, append=True):
        """
        :param word2vec_corpus: utils.Word2VecCorpus (like)
            It yield sent. The form of sent is list of str
        :param word_set: set of str
            Words that we want to infer vectors
        :param append: Boolean
            If True, vector of unseen words are stored in model.

        It returns
        ----------
        y : numpy.ndarray
            Inferred word vectors. (n_words, size)
        """

        WW, idx_to_vocab_ = self.vectorize_word_context_matrix(
            word2vec_corpus, word_set)

        self.infer_wordvec_from_vector(WW, idx_to_vocab_, append)

    def infer_wordvec_from_vector(self, X, row_to_vocab=None, append=True):
        """
        :param X: scipy.sparse.csr_matrix
            (word, context) cooccurrance matrix
        :param row_to_vocab: list of str
            Word index that corresponds row of X
        :param append: Boolean
            If True, vector of unseen words are stored in model.

        It returns
        ----------
        y : numpy.ndarray
            Inferred word vectors. (n_words, size)
        """

        if (append) and (row_to_vocab is None):
            raise ValueError('row_to_vocab should be inserted if append = True')

        pmi_ww, _, _ = train_pmi(X,
            py=self._py,  beta=1, min_pmi=0)

        y = safe_sparse_dot(pmi_ww, self._transformer)

        if append:
            n = self.wv.shape[0]
            idx_ = [i for i, vocab in enumerate(row_to_vocab)
                    if not (vocab in self._vocab_to_idx_)]

            # if exist no word to be appended
            if not idx_:
                return y

            vocabs_ = [row_to_vocab[i] for i in idx_]
            vec_ = y[np.asarray(idx_)]

            self._idx_to_vocab += vocabs_
            for i, vocab in enumerate(vocabs_):
                self._vocab_to_idx_[vocab] = n + i
            self.wv = np.vstack([self.wv, vec_])
            self.n_vocabs += len(idx_)

            if self._verbose:
                print('%d terms are appended' % len(vocabs_))

        return y

    def vectorize_word_context_matrix(self, word2vec_corpus, word_set):
        """
        :param word2vec_corpus: utils.Word2VecCorpus (like)
            It yield sent. The form of sent is list of str
        :param word_set: set of str
            Words that we want to infer vectors

        It returns
        ----------
        WW : scipy.sparse.csr_matrix
            WW[word, context] = frequency
        idx_to_vocab_ : list of str
            Word list that corresponds rows of WW
        """

        WWd = word_context(
            sents = word2vec_corpus,
            windows = self._window,
            dynamic_weight = self._dynamic_weight,
            verbose = self._verbose,
            vocab_to_idx = self._vocab_to_idx,
            row_vocabs = word_set
        )

        idx_to_vocab_ = [vocab for vocab in sorted(word_set)]
        vocab_to_idx_ = {vocab:idx for idx, vocab in enumerate(idx_to_vocab_)}

        WW = dict_to_sparse(
            dd = WWd,
            row_to_idx = vocab_to_idx_,
            col_to_idx = self._vocab_to_idx)

        return WW, idx_to_vocab_