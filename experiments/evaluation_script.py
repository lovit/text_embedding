import numpy as np
from evaluation import load_full_model
from evaluation import load_partial_model
from evaluation import load_index
from evaluation import most_similar

dataset_name = 'imdb'

directory = '../../../experiments/{}_type1/'.format(dataset_name)
queryset_directory = './{}_wordlist/'.format(dataset_name)
headers = ['{}_r0.001', '{}_r0.01', '{}_r0.05', '{}_r0.1', '{}_r0.2']
headers = [h.format(dataset_name) for h in headers]

test_vocabs_all = set()

for header in headers:
    print('begin {}'.format(header))
    wv, idx_to_vocab, vocab_to_idx = load_partial_model(directory, header)
    test_vocabs, _ = load_index('{}/{}.txt'.format(queryset_directory, header))
    test_vocabs = [q for q in test_vocabs if q in vocab_to_idx]
    test_vocabs_all.update(test_vocabs)

    query_idxs = np.asarray([vocab_to_idx[q] for q in test_vocabs])
    idxs, sims = most_similar(query_idxs, wv, topk=30)
    with open('{}.txt'.format(header), 'w', encoding='utf-8') as f:
        for query, idx, sim in zip(test_vocabs, idxs, sims):
            idx_sim = [(idx_to_vocab[idx_i], sim_i) for idx_i, sim_i in zip(idx, sim)]
            form = '{},{}'.format(query, idx_sim)
            f.write('{}\n'.format(form))
    print('done {}\n'.format(header))


wv, idx_to_vocab, vocab_to_idx = load_full_model(directory)
test_vocabs = list(test_vocabs_all)
query_idxs = np.asarray([vocab_to_idx[q] for q in test_vocabs])
idxs, sims = most_similar(query_idxs, wv, topk=30)
with open('{}_full_answer.txt'.format(dataset_hane), 'w', encoding='utf-8') as f:
    for query, idx, sim in zip(test_vocabs, idxs, sims):
        idx_sim = [(idx_to_vocab[idx_i], sim_i) for idx_i, sim_i in zip(idx, sim)]
        form = '{},{}'.format(query, idx_sim)
        f.write('{}\n'.format(form))
print('done full model\n')