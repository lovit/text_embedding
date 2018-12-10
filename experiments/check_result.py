import argparse
import glob
import numpy as np
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
    with open('%s/full_vocablist.txt' % result_directory, encoding='utf-8') as f:
        idx_to_vocab = [word.strip() for word in f if word.strip()]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}

    for path in wordlists:
        name = path.split('/')[-1][:-4]
        with open(path, encoding='utf-8') as f:
            wordset = [word.strip() for word in f][:3]
        wv_infer = np.loadtxt('%s/%s_wv.txt' % (result_directory, name))
        print('\\nnExperiment %s' % name)
        for word in wordset:
            compare(word, vocab_to_idx, idx_to_vocab, wv_full, wv_infer)
            print('-' * 50)

def compare(word, vocab_to_idx, idx_to_vocab, wv_full, wv_infer, topk=10):
    similars_full = similar_words(word, vocab_to_idx, idx_to_vocab, wv_full, topk)
    similars_infer = similar_words(word, vocab_to_idx, idx_to_vocab, wv_infer, topk)
    print('query = %s' % word)
    print('              FULL EMBEDDING\tINFER EMBEDDING')
    for sim_ful, sim_inf in zip(similars_full, similars_infer):
        print('%20s (%.3f)\t%s (%.3f)' % (sim_ful[0], sim_ful[1], sim_inf[0], sim_inf[1]))

if __name__ == '__main__':
    main()