from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

from .pmi import train_pmi
from .utils import get_process_memory
from .utils import check_dirs
from .vectorizer import sents_to_unseen_word_contexts_matrix
from .vectorizer import _scanning_vocabulary
from .vectorizer import _word_context
from .vectorizer import _encode_as_matrix

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
        self.n_vocabs = 0

        if sentences:
            self.train(sentences)

    @property
    def is_trained(self):
        return self.wv is not None

    def train(self, sentences):
        """
        Attributes
        ----------
        sentences : list of list of str (like)
            Iterable of iterables, optional
            A sentence is represented with list of str.
        """

        if self.is_trained:
            raise ValueError('Word2Vec model already trained')

        self._vocab2idx, self._idx2vocab = self._scan_vocabulary(sentences)
        self.n_vocabs = len(self._idx2vocab)

        X = self._vectorize_cooccurrance_matrix(sentences)

        pmi, _, self._py = self._train_pmi(X)

        self.wv, self._components = self._train_transform_svd(pmi)

    def _scan_vocabulary(self, sentences):
        vocab2idx, idx2vocab = _scanning_vocabulary(
            sentences,
            min_tf = self._min_count,
            tokenizer = self._tokenizer,
            verbose = self._verbose)
        return vocab2idx, idx2vocab

    def _vectorize_cooccurrance_matrix(self, sentences):
        word2contexts = _word_context(sentences,
            windows = self._window,
            tokenizer = self._tokenizer,
            dynamic_weight = self._dynamic_weight,
            verbose = self._verbose,
            vocab2idx = self._vocab2idx)
        X = _encode_as_matrix(word2contexts, self._vocab2idx, False)
        return X

    def _train_pmi(self, X):
        if self._verbose:
            print('Training PMI ...', end='', flush=True)

        pmi, px, py = train_pmi(X,
            alpha = self._alpha,
            beta = self._beta,
            as_csr = True,
            verbose = self._verbose)
        return pmi, px, py

    def _train_transform_svd(self, pmi):
        if self._verbose:
            print(' done\nTraining SVD ...', end='', flush=True)

        U, S, VT = fit_svd(pmi, self._size, self._n_iter)
        S_ = S ** (1/2)
        wordvec = U * S_
        contextvec = VT.T * S_
        return wordvec, contextvec

    def infer(self, sentences, words, append=True, tokenizer=None):
        """
        Attributes
        ----------
        sentences : list of list of str (like)
            Iterable of iterables, optional
            A sentence is represented with list of str.
        words : list or set of str
            Word set of that we want to infer vectors.
        append : Boolean
            If true, the inferring results are stored in Word2Vec model.
        tokenizer : functional
            Tokenizer functions. It assumes that the input form is str
            and output form is list of str
        """

        if tokenizer is None:
            tokenizer = self._tokenizer

        # create (word, contexts) matrix
        x_, idx2vocab_ = sents_to_unseen_word_contexts_matrix(
            sentences, words, self._vocab2idx)

        if self._verbose:
            print('Training PMI ...', end='', flush=True)

        # infer pmi
        pmi_, _, _ = train_pmi(x_, py=self._py,
            alpha=self._alpha, beta=1.0, as_csr=True, verbose=True)

        if self._verbose:
            print(' done\nApplying trained SVD ...', end='', flush=True)

        # apply trained SVD
        y_ = safe_sparse_dot(pmi_, self._components)

        if self._verbose:
            print(' done', flush=True)

        if append:
            if self._verbose:
                print('vocabs : {} -> '.format(self.n_vocabs), end='', flush=True)

            self._idx2vocab += idx2vocab_
            self._vocab2idx.update({vocab : idx + self.n_vocabs
                for idx, vocab in enumerate(idx2vocab_)})
            self.wv = np.vstack([self.wv, y_])
            self.n_vocabs += len(idx2vocab_)

            if self._verbose:
                print('{}'.format(self.n_vocabs), flush=True)

        return y_, idx2vocab_

    def most_similar(self, word, topk=10):
        """
        Attributes
        ----------
        word : str
            Query word
        topk : int
            Number of most similar words

        It returns
        ----------
        similars : list of tuple
            List of tuple of most similar words.
            Each tuple consists with (word, cosine similarity)
        """

        query_idx = self._vocab2idx.get(word, -1)

        if query_idx < 0:
            return []

        query_vector = self.wv[query_idx,:].reshape(1,-1)
        return self.most_similar_from_vector(query_vector, topk, query_idx)

    def most_similar_from_vector(self, vector, topk=10, query_idx=-1):
        """
        Attributes
        ----------
        vector : numpy.ndarray
            A vector of query word. Its shape should be (1, self._size)
        topk : int
            Number of most similar words

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
            similar_word = self._idx2vocab[similar_idx]
            similars.append((similar_word, 1-dist[similar_idx]))

        return similars

    def save(self, path):
        raise NotImplemented


def fit_svd(X, n_components, n_iter=5, random_state=None):

    if (random_state == None) or isinstance(random_state, int):
        random_state = check_random_state(random_state)

    n_features = X.shape[1]

    if n_components >= n_features:
        raise ValueError("n_components must be < n_features;"
                         " got %d >= %d" % (n_components, n_features))

    U, Sigma, VT = randomized_svd(
        X, n_components,
        n_iter = n_iter,
        random_state = random_state)

    return U, Sigma, VT