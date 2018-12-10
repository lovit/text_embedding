import argparse
import glob
import os
import sys

try:
    sys.path.append('../')
    import numpy as np
    import text_embedding
    from text_embedding import Word2Vec
    from text_embedding import Word2VecCorpus
    from text_embedding import WordVectorInferenceDecorator
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

    print('Word vector inference test (Learn from WordVectorInferenceDecorator)')
    print('debug mode is {} / verbose mode is {}'.format(debug, verbose))

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    wordlists = glob.glob('%s/*txt' % wordlist_directory)
    print('num of inference wordset = %d' % len(wordlists))

    num_doc = 10000 if debug else -1
    corpus = Word2VecCorpus(corphs_path, num_doc = num_doc, sentence_separator='  ')

    # train full model
    word2vec = Word2Vec(corpus, size=size, window=window, min_count=min_count,
        negative=negative, alpha=alpha, beta=beta, dynamic_weight=dynamic_weight,
        verbose=verbose, n_iter=n_iter, min_cooccurrence=min_cooccurrence, prune_point=300000)

    save_model(word2vec, result_directory, 'full')

    # train infer model
    for i, path in enumerate(wordlists):
        # load test terms
        with open(path, encoding='utf-8') as f:
            wordset = {word.strip() for word in f if word.strip()}
        exp_name = path.split('/')[-1][:-4]
        # train base model
        inference_corpus = WordVectorInferenceDecorator(corpus, wordset, training=True)
        word2vec = Word2Vec(inference_corpus, size=size, window=window, min_count=min_count,
            negative=negative, alpha=alpha, beta=beta, dynamic_weight=dynamic_weight,
            verbose=verbose, n_iter=n_iter, min_cooccurrence=min_cooccurrence, prune_point=300000)
        # inference
        inference_corpus.training = False
        infered_vec = word2vec.infer_wordvec(inference_corpus, wordset, append=True)
        # save model
        save_model(word2vec, result_directory, exp_name)

        print('%d / %d done with %s' % (i+1, len(wordlists), exp_name))

def save_model(model, directory, header):
    model_path = '%s/%s_wv.txt' % (directory, header)
    vocab_path = '%s/%s_vocab.txt' % (directory, header)
    np.savetxt(model_path, model.wv)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for vocab in model._idx_to_vocab:
            f.write('%s\n' % vocab)

if __name__ == '__main__':
    main()