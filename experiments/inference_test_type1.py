import argparse
import glob
import os
import sys

try:
    sys.path.append('../')
    import numpy as np
    import text_embedding
    from text_embedding import scan_vocabulary
    from text_embedding import word_context
    from text_embedding import dict_to_sparse
    from text_embedding import train_pmi
    from text_embedding import train_svd
    from text_embedding import Word2VecCorpus
    from sklearn.utils.extmath import safe_sparse_dot
    from scipy.io import mmwrite
    print('Library was imported successfully')
except Exception as e:
    print(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, help='text corpus', required=True)
    parser.add_argument('--wordlist_directory', type=str, help='root directory inference word lists', required=True)
    parser.add_argument('--result_directory', type=str, help='result directory', required=True)
    parser.add_argument('--size', type=int, default=300, help='embedding dimension')
    parser.add_argument('--window', type=int, default=4, help='context window size')
    parser.add_argument('--min_count', type=int, default=20, help='min count of words')
    parser.add_argument('--negative', type=int, default=10, help='number of negative samples')
    parser.add_argument('--alpha', type=float, default=0.0, help='pmi smoothing factor')
    parser.add_argument('--beta', type=float, default=0.75, help='pmi smoothing factor')
    parser.add_argument('--no-dynamic-weight', dest='dynamic_weight', action='store_false', help='no use dynamic weight')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='no verbose mode')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='no debug mode')
    parser.add_argument('--n_iter', type=int, default=5, help='number of iteration of svd algorithm')
    parser.add_argument('--min_cooccurrence', type=int, default=2, help='min count of cooccurrence')

    args = parser.parse_args()
    corphs_path = args.corpus_path
    wordlist_directory = args.wordlist_directory
    result_directory = args.result_directory
    size = args.size
    window = args.window
    min_count = args.min_count
    negative = args.negative
    alpha = args.alpha
    beta = args.beta
    dynamic_weight = args.dynamic_weight
    verbose = args.verbose
    debug = args.debug
    n_iter = args.n_iter
    min_cooccurrence = args.min_cooccurrence

    print('Word vector inference test (Learn from co-occurrence matrix)')
    print('debug mode is {} / verbose mode is {}'.format(debug, verbose))

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    wordlists = glob.glob('%s/*txt' % wordlist_directory)
    print('num of inference wordset = %d' % len(wordlists))

    num_doc = 10000 if debug else -1
    corpus = Word2VecCorpus(corphs_path, num_doc = num_doc, sentence_separator='  ')

    # train full model
    idx_to_vocab, vocab_to_idx, idx_to_count, WW, wv = train_full_model(
        corpus, min_count, verbose, window, min_cooccurrence,
        dynamic_weight, beta, negative, size, n_iter)
    np.savetxt('%s/full_wv.txt' % result_directory, wv)
    mmwrite('%s/full_WW.mtx' % result_directory, WW)
    with open('%s/full_vocablist.txt' % result_directory, 'w', encoding='utf-8') as f:
        for vocab in idx_to_vocab:
            f.write('%s\n'%vocab)
    with open('%s/full_vocabcount.txt' % result_directory, 'w', encoding='utf-8') as f:
        for count in idx_to_count:
            f.write('{}\n'.format(count))

    # train inference
    for i, path in enumerate(wordlists):
        with open(path, encoding='utf-8') as f:
            wordset = {word.strip() for word in f if word.strip()}

        exp_name = path.split('/')[-1][:-4]
        wv = train_infer_model(idx_to_vocab, vocab_to_idx, idx_to_count,
            WW, wordset, window, dynamic_weight, beta, negative, size, n_iter)

        result_path = '%s/%s_wv.txt' % (result_directory, exp_name)
        np.savetxt(result_path, wv)
        print('%d / %d done with %s' % (i+1, len(wordlists), exp_name))

def train_full_model(corpus, min_count, verbose, window,
    min_cooccurrence, dynamic_weight, beta, negative, size, n_iter):

    vocab_to_idx, idx_to_vocab, idx_to_count = scan_vocabulary(
        corpus, min_count=min_count, verbose=verbose)

    WW = _make_word_context_matrix(corpus, window, min_cooccurrence,
        dynamic_weight, verbose, vocab_to_idx)

    py = np.asarray(idx_to_count)

    pmi_ww, _, py = train_pmi(WW, beta = beta,
        py = py, min_pmi = np.log(max(1, negative)))

    if verbose:
        print('train SVD ... ', end='')

    wv, transformer = _get_repr_transformer(pmi_ww, size, n_iter)

    if verbose:
        print('done')

    return idx_to_vocab, vocab_to_idx, idx_to_count, WW, wv

def train_infer_model(idx_to_vocab, vocab_to_idx, idx_to_count, WW, wordset, window, dynamic_weight, beta, negative, size, n_iter):

    n_vocabs = len(idx_to_vocab)

    wordset = set(wordset)
    word_idxs = {vocab_to_idx[vocab] for vocab in wordset if vocab in vocab_to_idx}

    train_idxs = np.asarray([idx for idx, vocab in enumerate(idx_to_vocab) if not (vocab in wordset)])
    test_idxs = np.asarray([idx for idx, vocab in enumerate(idx_to_vocab) if vocab in wordset])

    py_train = np.asarray(idx_to_count)[train_idxs]
    WW_train = WW[train_idxs][:,train_idxs]
    WW_test = WW[test_idxs][:,train_idxs]

    pmi_train, _, py = train_pmi(WW_train, beta = beta,
        py = py_train, min_pmi = 0)
    wv_base, transformer = _get_repr_transformer(pmi_train, size, n_iter)

    pmi_test, _, _ = train_pmi(WW_test, beta = 1,
        py = py, min_pmi = np.log(max(1, negative)))
    wv_infer = safe_sparse_dot(pmi_test, transformer)

    wv = np.zeros((n_vocabs, size))
    wv[train_idxs] = wv_base
    wv[test_idxs] = wv_infer

    return wv

def _make_word_context_matrix(corpus, window, min_cooccurrence,
    dynamic_weight, verbose, vocab_to_idx, prune_point=300000):

    WWd = word_context(
        sents = corpus,
        windows = window,
        min_count = min_cooccurrence,
        dynamic_weight = dynamic_weight,
        verbose = verbose,
        vocab_to_idx = vocab_to_idx,
        prune_point = prune_point
    )

    WW = dict_to_sparse(
        dd = WWd,
        row_to_idx = vocab_to_idx,
        col_to_idx = vocab_to_idx)

    return WW

def _get_repr_transformer(X, size, n_iter):
    U, S, VT = train_svd(X, n_components=size, n_iter=n_iter)
    S_ = S ** (0.5)
    wv = U * S_
    transformer = VT.T * (S ** (-1))
    return wv, transformer

if __name__ == '__main__':
    main()