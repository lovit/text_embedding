from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

from .pmi import train_pmi
from .utils import get_process_memory
from .utils import check_dirs
from .vectorizer import sents_to_word_contexts_matrix
from .vectorizer import sents_to_unseen_word_contexts_matrix

class Word2Vec:

    def __init__(self, sentences=None, size=100, window=3, min_count=10,
        negative=10, alpha=0.0001, tokenizer=lambda x:x.split(), verbose=True):

        """
        Attributes
        ----------
        sentences : list of list of str (like)
            Iterable of iterables, optional
            A sentence is represented with list of str.
        size : int
            Word vector dimension
            Default is 100
        window : int
            The number of context words is 2 x window
            Default is 3
        min_count : int
            Minumum frequency of words
            Default is 10
        negative : int
            Number of negative samples. Minimum PMI is automatically
            defined with this value; log(negative)
            Default is 10
        alpha : float
            Nonnegative, PMI smoothing factor
            Default is 0.0001
        """

        # user defined parameters
        self._size = size
        self._window = window
        self._min_count = min_count
        self._negative = negative
        self._alpha = alpha
        self._tokenizer = tokenizer
        self._verbose = verbose

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

        x, self._idx2vocab = sents_to_word_contexts_matrix(
            sentences, self._window, self._min_count,
            self._tokenizer, self._verbose)

        self._vocab2idx = {vocab:idx for idx, vocab
            in enumerate(self._idx2vocab)}
        self.n_vocabs = len(self._idx2vocab)

        pmi, self._py = train_pmi(x, alpha=self._alpha,
            as_csr=True, verbose=self._verbose)

        svd = TruncatedSVD(n_components=self._size)
        self.wv = svd.fit_transform(pmi)
        self._components = svd.components_
        self._explained_variance = svd.explained_variance_
        self._explained_variance_ratio = svd.explained_variance_ratio_

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

        # infer pmi
        pmi_, _ = train_pmi(x_, py=self._py,
            alpha=self._alpha, as_csr=True, verbose=True)

        # apply trained SVD
        y_ = safe_sparse_dot(pmi_, self._components.T)

        if append:
            self._idx2vocab += idx2vocab_
            self._vocab2idx.update({vocab : idx + self.n_vocabs
                for idx, vocab in enumerate(idx2vocab_)})
            self.wv = np.vstack([self.wv, y_])
            self.n_vocabs += len(idx2vocab_)

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