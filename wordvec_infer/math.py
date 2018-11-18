import numpy as np
import math
from scipy.sparse import diags
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import safe_sparse_dot

def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag

def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data > min_exp_pmi)[0]

    # apply logarithm
    exp_pmi.data = np.log(exp_pmi.data)
    return exp_pmi

def train_pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """
    :param X: scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    :param py: numpy.ndarray
        (1, word) shape, probability of context words.
    :param min_pmi: float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    :param alpha: float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0
    :param beta: float
        Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
        Default is 1.0

    It returns
    ----------
    pmi : scipy.sparse.dok_matrix or scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)
    """

    assert 0 < beta <= 1

    # convert x to probability matrix & marginal probability 
    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    pxy = X / X.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py

def fit_svd(X, n_components, n_iter=5, random_state=None):
    """
    :param X: scipy.sparse.csr_matrix
        Input matrix
    :param n_components: int
        Size of embedding dimension
    :param n_iter: int
        Maximum number of iteration. Default is 5
    :param random_state: random state
        Default is None

    It returns
    ----------
    U : numpy.ndarray
        Representation matrix of rows. shape = (n_rows, n_components)
    Sigma : numpy.ndarray
        Eigenvalue of dimension. shape = (n_components, n_components)
        Diagonal value are in decreasing order
    VT : numpy.ndarray
        Representation matrix of columns. shape = (n_components, n_cols)
    """

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

def compute_embedding_difference(w2v_transformer, d2v_transformer, pmi_ww,
    batch_size=1000, topk=100, verbose=True):

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
