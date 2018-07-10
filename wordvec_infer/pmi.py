import numpy as np
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from .utils import get_process_memory

def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1/(v + alpha) for v in px_diag.data[0]])
    return px_diag

def _as_csr_matrix(exp_pmi, min_pmi, verbose):
    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data > min_exp_pmi)[0]

    # apply logarithm
    exp_pmi.data = np.log(exp_pmi.data)

    return exp_pmi

def _as_dok_matrix(exp_pmi, min_pmi, verbose):
    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data > min_exp_pmi)[0]

    pmi_dok = dok_matrix(exp_pmi.shape)

    # prepare data (rows, cols, data)
    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    # enumerate function for printing status
    for _n_idx, idx in enumerate(indices):

        # print current status
        if verbose and _n_idx % 10000 == 0:
            print('\rcomputing pmi {:.3} %  mem={} Gb    '.format(
                100 * _n_idx / indices.shape[0], '%.3f' % get_process_memory())
                  , flush=True, end='')

        # apply logarithm
        pmi_dok[rows[idx], cols[idx]] = np.log(data[idx])

    if verbose:
        print('\rcomputing pmi was done{}'.format(' '*30), flush=True)

    return pmi_dok

def train_pmi(x, py=None, min_pmi=0, alpha=0.0001, as_csr=True, verbose=False):
    """
    Attributes
    ----------
    x : scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    min_pmi : float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    alpha : float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0001
    as_csr : Boolean
        Return type is csr_matrix if as_csr is True else dok_matrix
    verbose : Boolean
        Print progress if verbose is true.
        Default is False

    It returns
    ----------
    pmi : scipy.sparse.dok_matrix or scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of words
    """

    # convert x to probability matrix & marginal probability 
    px = (x.sum(axis=1) / x.sum()).reshape(-1)
    if py is None:
        py = (x.sum(axis=0) / x.sum()).reshape(-1)
    pxy = x / x.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    if as_csr:
        pmi = _as_csr_matrix(exp_pmi, min_pmi, verbose)
    else:
        pmi = _as_dok_matrix(exp_pmi, min_pmi, verbose)

    return pmi, px