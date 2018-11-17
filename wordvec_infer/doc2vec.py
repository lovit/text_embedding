import numpy as np
import scipy as sp
from .math import fit_svd
from .math import train_pmi
from .vectorizer import dict_to_sparse
from .vectorizer import label_word
from .vectorizer import scan_vocabulary
from .word2vec import Word2Vec

class Doc2Vec(Word2Vec):
    def __init__(self, sentences=None, size=100, window=3, min_count=10,
        negative=10, alpha=0.0, beta=0.75, dynamic_weight=False,
        verbose=True, n_iter=5):

        super().__init__(sentences, size, window,
            min_count, negative, alpha, beta,
            dynamic_weight, verbose, n_iter)

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
        self._vocab_to_idx, self._idx_to_vocab = scan_vocabulary(
            doc2vec_corpus, min_count=self._min_count)
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
        self.dv = representation[n_vocab:]
        self._transformer = transformer[:n_vocab]
        self.n_vocabs = n_vocab

        if self._verbose:
            print('done')

        self._transformer_ = self._get_word2vec_transformer(WW)
        self._label_influence = self._get_label_influence(
            self._transformer, self._transformer_)

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
        pmi_ww, _, _ = train_pmi(WW, beta=self._beta, min_pmi=0)
        _, transformer = self._get_repr_and_trans(pmi_ww)
        return transformer

    def _get_label_influence(self, u, v):
        # TODO
        return None

    def similar_docs(self, bow, topk=10):
        raise NotImplemented

    def infer_docvec(self, bow):
        raise NotImplemented