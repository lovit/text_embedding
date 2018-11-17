from .math import train_pmi
from .math import fit_svd
from .vectorizer import scan_vocabulary
from .vectorizer import word_context
from .vectorizer import dict_to_sparse
from sklearn.metrics import pairwise_distances

class Word2Vec:

    def __init__(self, sentences=None, size=100, window=3, min_count=10,
        negative=10, alpha=0.0, beta=0.75, dynamic_weight=False, verbose=True,
        n_iter=5, tokenizer=lambda x:x):

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
        :param tokenizer: callable
            Default is lambda x:x
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
        self._tokenizer = tokenizer
        self._n_iter = n_iter

        # trained attributes
        self.wv = None # word vector
        self._vocab2idx = None
        self._idx2vocab = None
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
        Attributes
        ----------
        sentences : list of list of str (like)
            Iterable of iterables, optional
            A sentence is represented with list of str.
        """

        if self.is_trained:
            raise ValueError('Word2Vec model already trained')

        self._vocab_to_idx, self._idx_to_vocab = scan_vocabulary(
            word2vec_corpus, min_count=10)

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

        U, S, VT = fit_svd(pmi_ww, n_components=100, n_iter=5)
        S_ = S ** (0.5)
        self.wv = U * S_
        self._transformer = VT.T * (S_ ** (-1))

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

        query_idx = self._vocab_to_idx.get(word, -1)

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