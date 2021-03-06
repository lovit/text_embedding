__version__ = '0.0.1'

from .doc2vec import Doc2Vec
from .math import train_svd
from .math import train_pmi
from .utils import get_process_memory
from .utils import check_dirs
from .utils import most_similar
from .utils import Word2VecCorpus
from .utils import Doc2VecCorpus
from .utils import WordVectorInferenceDecorator
from .vectorizer import scan_vocabulary
from .vectorizer import dict_to_sparse
from .vectorizer import word_context
from .vectorizer import label_word
from .word2vec import Word2Vec
