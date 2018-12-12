import argparse
import copy
import glob
import numpy as np
import os
from utils import similar_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_directory', type=str, help='result directory', required=True)
    parser.add_argument('--wordlist_directory', type=str, help='root directory inference word lists', required=True)
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='no debug mode')

    args = parser.parse_args()
    wordlist_directory = args.wordlist_directory
    result_directory = args.result_directory
    debug = args.debug

    wordlists = glob.glob('%s/*txt' % wordlist_directory)
    print('num of inference wordset = %d' % len(wordlists))

    wv_full = np.loadtxt('%s/full_wv.txt' % result_directory)
    with open('%s/full_vocab.txt' % result_directory, encoding='utf-8') as f:
        idx_to_vocab = [word.strip() for word in f if word.strip()]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    with open('%s/full_vocabcount.txt' % result_directory, encoding='utf-8') as f:
        idx_to_count = [int(word.strip()) for word in f if word.strip()]

    count = lambda w:idx_to_count[vocab_to_idx[w]]

    if debug:
        wordlists = wordlists[:3]

    for path in wordlists:
        name = path.split('/')[-1][:-4]
        with open(path, encoding='utf-8') as f:
            wordset = [word.strip() for word in f]
            wordset = [word for word in wordset if word in vocab_to_idx]
            wordset = sorted(wordset, key=lambda w:-count(w))[:1000]
            wordset = np.random.permutation(wordset).tolist()[:10]
        wv_infer = np.loadtxt('%s/%s_wv.txt' % (result_directory, name))
        if os.path.exists('%s/%s_vocab.txt' % (result_directory, name)):
            vocab_to_idx_, idx_to_vocab_, wv_base, = remain_only_common_vocabs(
                wv_full, vocab_to_idx, '%s/%s_vocab.txt' % (result_directory, name))
        else:
            wv_base = wv_full.copy()
            vocab_to_idx_ = copy.deepcopy(vocab_to_idx)
            idx_to_vocab_ = copy.deepcopy(idx_to_vocab)
        print('\n\nExperiment %s' % name)
        for word in wordset:
            compare(word, vocab_to_idx_, idx_to_vocab_, wv_base, wv_infer, count)
            print('-' * 50)

def remain_only_common_vocabs(wv_full, vocab_to_idx_full, vocab_path_infer):
    with open(vocab_path_infer, encoding='utf-8') as f:
        idx_to_vocab = [vocab.strip() for vocab in f if vocab.strip()]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    full_idxs = np.asarray([vocab_to_idx_full[vocab] for vocab in idx_to_vocab])
    wv_base = wv_full[full_idxs]
    return vocab_to_idx, idx_to_vocab, wv_base

def compare(word, vocab_to_idx, idx_to_vocab, wv_full, wv_infer, count_func, topk=10):
    similars_full = similar_words(word, vocab_to_idx, idx_to_vocab, wv_full, topk)
    similars_infer = similar_words(word, vocab_to_idx, idx_to_vocab, wv_infer, topk)
    print('query = %s (%d)' % (word, count_func(word)))
    print('              FULL EMBEDDING      \tINFER EMBEDDING')
    for sim_ful, sim_inf in zip(similars_full, similars_infer):
        print('%20s (%6d, %.3f)\t%s (%6d, %.3f)' % (sim_ful[0], count_func(sim_ful[0]), sim_ful[1], sim_inf[0], count_func(sim_inf[0]), sim_inf[1]))

if __name__ == '__main__':
    main()