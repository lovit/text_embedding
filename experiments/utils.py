from sklearn.metrics import pairwise_distances

def similar_words(word, vocab_to_idx, idx_to_vocab, wv, topk=10):
    query_idx = vocab_to_idx.get(word, -1)
    if query_idx < 0:
        return []

    query_vector = wv[query_idx,:].reshape(1,-1)
    dist = pairwise_distances(query_vector, wv, metric='cosine')[0]

    similars = []
    for similar_idx in dist.argsort():
        if similar_idx == query_idx:
            continue
        if len(similars) >= topk:
            break
        similar_word = idx_to_vocab[similar_idx]
        similars.append((similar_word, 1-dist[similar_idx]))

    return similars