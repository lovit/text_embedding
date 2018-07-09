from .pmi import train_pmi
from .pmi import infer_pmi
from .utils import get_process_memory
from .utils import check_dirs
from .vectorizer import sent_to_word_contexts_matrix

class Word2Vec:

    def __init__(self, sentences=None, size=100, window=3, 
                 min_count=10, negative=10, alpha=0.0001, verbose=True):
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
        self._verbose = verbose

        # trained attributes
        self.wv = None # word vector
        self._vocab2int = None
        self._int2vocab = None
        self._transformer = None

        if sentences:
            self.train(sentences)

        @property
        def is_trained(self):
            return self.wv is not None

        def train(self, sentences):
            raise NotImplemented

        def infer(self, sentences, words):
            raise NotImplemented

        def save(self, path):
            raise NotImplemented