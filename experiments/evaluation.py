import os
import numpy as np
from sklearn.metrics import pairwise_distances


def load_model(vec_path):
    return np.loadtxt(vec_path)

def load_index(index_path):
    with open(index_path, encoding='utf-8') as f:
        idx_to_vocab = [vocab.strip() for vocab in f]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def load_full_model(directory):
    vocab_path = '{}/full_vocab.txt'.format(directory)
    vec_path = '{}/full_wv.txt'.format(directory)
    idx_to_vocab, vocab_to_idx = load_index(vocab_path)
    wv = load_model(vec_path)
    return wv, idx_to_vocab, vocab_to_idx

def load_partial_model(directory, header):
    vocab_path = '{}/{}_vocab.txt'.format(directory, header)
    if not os.path.exists(vocab_path):
        vocab_path = '{}/full_vocab.txt'.format(directory)
    vec_path = '{}/{}_wv.txt'.format(directory, header)
    idx_to_vocab, vocab_to_idx = load_index(vocab_path)
    wv = load_model(vec_path)
    return wv, idx_to_vocab, vocab_to_idx

def most_similar(query_idxs, X, topk, batch_size=100):
    n_queries = query_idxs.shape[0]
    n_batch = math.ceil(n_queries / batch_size)
    idxs = []
    sims = []
    for i_batch in range(n_batch):
        b = i_batch * batch_size
        e = min(n_queries, (i_batch + 1) * batch_size)
        query_idxs_ = query_idxs[b:e]
        idxs_, sims_ = _most_similar(query_idxs_, X, topk)
        idxs.append(idxs_)
        sims.append(sims_)
        print('\rsimilarity computation {} / {} batch ... '.format(i_batch + 1, n_batch), end='')
    print('\rsimilarity computation {0} / {0} batch was done'.format(n_batch))

    idxs = np.vstack(idxs)
    sims = np.vstack(sims)
    return idxs, sims

def _most_similar(query_idxs, X, topk):
    query_vec = X[query_idxs]
    dist = pairwise_distances(query_vec, X, metric='cosine')
    dist_ = dist.copy()
    dist_.sort(axis=1)
    similar_idx = dist.argsort(axis=1)
    similar_idx = similar_idx[:,1:topk+1]
    similarity = 1 - dist_[:,1:topk+1]
    return similar_idx, similarity

def compare(wv, wv_, vocab_to_idx, vocab_to_idx_, test_vocabs, topk=10):
    """
    wv : numpy.ndarray
        Embedding vector of full model with shape = (n, dim)
    wv_ : numpy.ndarray
        Embedding vector of inference model with shape (m, dim)
        m is smaller or equal with n
    vocab_to_idx : dict
        Dictionary term to idx. len == n
    vocab_to_idx_ : dict
        Dictionary term to idx. len = m
        Subset of vocab_to_idx, but different index

    Returns
    -------
    similars : numpy.ndarray
        Top k most similar term idxs in full model
    similars_ : numpy.ndarray
        Top k most similar term idxs in inference model
    idx_to_vocab : list of str
        Index of vocabs. 
    """

    intersections = {vocab for vocab in vocab_to_idx if vocab in vocab_to_idx_}
    intersections = {vocab:idx for idx, vocab in enumerate(sorted(intersections))}
    idx_to_vocab = [vocab for vocab in sorted(intersections)]
    test_vocabs = {vocab for vocab in test_vocabs if vocab in intersections}
    query_idxs = np.asarray([intersections[vocab] for vocab in sorted(test_vocabs, key=lambda x:intersections[x])])

    indices = np.asarray([vocab_to_idx[vocab] for vocab in sorted(intersections)])
    indices_ = np.asarray([vocab_to_idx_[vocab] for vocab in sorted(intersections)])
    X = wv[indices]
    X_ = wv_[indices_]

    #similars = _most_similar(query_idxs, X)
    #similars_ = _most_similar(query_idxs, X_)

    #return similars, similars_, idx_to_vocab
    raise NotImplemented