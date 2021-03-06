import numpy as np

def proportion_keywords(S, min_score=0.5, topk=30,
    candidates_topk=100, index2word=None, is_stopword=None):

    l1_normalize = lambda x:np.asarray(x/x.sum())

    n_labels, n_terms = S.shape
    total = np.asarray(S.sum(axis=0)).reshape(-1)

    keywords = []
    for l in range(n_labels):
        n_prop = l1_normalize(total - S[l].reshape(-1))
        p_prop = l1_normalize(S[l,:].reshape(-1))

        indices = np.where(p_prop > 0)[0]
        indices = sorted(indices, key=lambda idx:-p_prop[idx])[:candidates_topk]
        scores = [(idx, p_prop[idx] / (p_prop[idx] + n_prop[idx])) for idx in indices]
        scores = [t for t in scores if t[1] >= min_score]
        scores = sorted(scores, key=lambda x:-x[1])
        keywords.append(scores)

    if index2word is not None:
        keywords = [[(index2word[idx], score) for idx, score in keyword] for keyword in keywords]

    if is_stopword is not None:
        keywords = [[(w,s) for w,s in keyword if not is_stopword(w)] for keyword in keywords]

    if topk > 0:
        keywords = [keyword[:topk] for keyword in keywords]

    return keywords