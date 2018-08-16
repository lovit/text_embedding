import os
from collections import defaultdict
from scipy.sparse import csr_matrix

from .utils import get_process_memory

def sents_to_word_contexts_matrix(sents, windows=3, min_tf=10,
        tokenizer=lambda x:x.split(), dynamic_weight=False, verbose=True):

    """
    See Improving distributional similarity with lessons learned from word embeddings,
    Omer Levy, Yoav Goldberg, and Ido Dagan, ACL 2015 for detail of dynamic_weight
    """
    if verbose:
        print('Create (word, contexts) matrix')

    vocab2idx, idx2vocab = _scanning_vocabulary(
        sents, min_tf, tokenizer, verbose)

    word2contexts = _word_context(
        sents, windows, tokenizer, dynamic_weight, verbose, vocab2idx)

    x = _encode_as_matrix(word2contexts, vocab2idx, verbose)

    if verbose:
        print('  - done')
    return x, idx2vocab

def sents_to_unseen_word_contexts_matrix(sents, unseen_words, vocab2idx,
    windows=3, min_tf=10, tokenizer=lambda x:x.split(), dynamic_weight=False, verbose=True):

    if verbose:
        print('Create (unseen word, contexts) matrix')

    _, idx2vocab_ = _scanning_vocabulary(
        sents, min_tf, tokenizer, verbose)

    # indices of unseen words that will be inferred.
    idx2vocab_ = [vocab for vocab in idx2vocab_
                  if (vocab in unseen_words) and not (vocab in vocab2idx)]

    n_vocabs_before = len(vocab2idx)
    vocab2idx_ = {vocab:idx + n_vocabs_before for idx, vocab
                  in enumerate(idx2vocab_)}

    word2contexts = _word_context(
        sents, windows, tokenizer, dynamic_weight, verbose,
        vocab2idx, base_vocabs = vocab2idx_)

    # merge two word indexs
    vocab2idx_merge = {vocab:idx for vocab, idx in vocab2idx.items()}
    vocab2idx_merge.update(vocab2idx_)

    x = _encode_as_matrix(word2contexts, vocab2idx_merge, verbose)

    # re-numbering rows; idx - n_vocabs_before
    x = _renumbering_rows(
        x, n_vocabs_before, n_rows=len(idx2vocab_), n_cols=n_vocabs_before)

    if verbose:
        print('  - done')
    return x, idx2vocab_

def _scanning_vocabulary(sents, min_tf, tokenizer, verbose):

    # counting word frequency, first
    word_counter = defaultdict(int)

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - counting word frequency', i_sent)

        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1

    # filtering with min_tf    
    vocab2idx = {word for word, count in word_counter.items() if count >= min_tf}
    vocab2idx = {word:idx for idx, word in enumerate(
        sorted(vocab2idx, key=lambda w:-word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocab2idx.items(), key=lambda w:w[1])]

    if verbose:
        _print_status('  - counting word frequency', i_sent)
        print(' #vocabs = {}'.format(len(vocab2idx)))

    del word_counter

    return vocab2idx, idx2vocab

def _print_status(message, i_sent, new_line=False):
    print('\r{} from {} sents, mem={} Gb'.format(
            message, i_sent, '%.3f' % get_process_memory()),
        flush=True, end='\n' if new_line else ''
    )

def _word_context(sents, windows, tokenizer,
    dynamic_weight, verbose, vocab2idx, base_vocabs=None):

    """
    Attributes
    ----------
    vocab2idx : dict {str:int}
        Index of context words
    base_vocabs : dict or set
        Index of base words
    """

    if not base_vocabs:
        base_vocabs = {vocab for vocab in vocab2idx}

    # scanning (word, context) pairs
    word2contexts = defaultdict(lambda: defaultdict(int))

    if dynamic_weight:
        weight = [(windows-i)/windows for i in range(windows)]
    else:
        weight = [1] * windows

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - scanning (word, context) pairs', i_sent)

        words = tokenizer(sent)
        if not words:
            continue

        n = len(words)

        for i, word in enumerate(words):
            if not (word in base_vocabs):
                continue

            # left_contexts
            for w in range(windows):
                j = i - (w + 1)
                if j < 0 or not (words[j] in vocab2idx):
                    continue
                word2contexts[word][words[j]] += weight[w]

            # right_contexts
            for w in range(windows):
                j = i + w + 1
                if j >= n or not (words[j] in vocab2idx):
                    continue
                word2contexts[word][words[j]] += weight[w]

    if verbose:
        _print_status('  - scanning (word, context) pairs', i_sent, new_line=True)

    return word2contexts

def _encode_as_matrix(word2contexts, vocab2idx, verbose):

    rows = []
    cols = []
    data = []
    for word, contexts in word2contexts.items():
        word_idx = vocab2idx[word]
        for context, cooccurrence in contexts.items():
            context_idx = vocab2idx[context]
            rows.append(word_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        print('  - (word, context) matrix was constructed. shape = {}{}'.format(
            x.shape, ' '*20))

    return x

def _renumbering_rows(x, n_vocabs, n_rows, n_cols):
    rows, cols = x.nonzero()
    data = x.data
    rows = rows - n_vocabs
    x_ = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return x_