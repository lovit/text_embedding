## Word vector inference

Word2Vec 은 words 를 크기가 정해진 공간 안의 distributed representation 으로 표현합니다. Skip-gram with Negative Sampling (SGNS) ^1 는 Word2Vec 의 구조 중 하나로, Softmax 의 효율적인 학습을 위하여 negative samplings 을 이용합니다.

Levy and Goldberg (2014) 는 ^2 에서 SGNS 는 (word, context) matrix 에 Shifted Positive Point Mutual Information (SPPMI) 를 적용한 결과임을 수학적, 실험적으로 증명하였습니다. PMI 는 informative contexts 를 선택하는 역할을 하며, Singular Value Decomposition (SVD) 는 유용한 문맥만으로 표현된 (word, context) matrix 를 정해진 크기의 공간으로 축소합니다.

학습데이터에 대하여 PMI 를 계산할 때 이용하는 context words probability 와 SVD 의 components 를 기억한다면, 새로운 단어에 대해서도 (word, context) matrix 를 만든 뒤, 동일한 PMI, SVD 를 적용하여, 학습데이터와 동일한 semantic space 의 word vector 로 새로운 단어들의 distributed representation 을 추정할 수 있습니다.

이 프로젝트는 PMI 와 SVD 를 이용하여 word (n-gram, phrase, doc and other entities) 를 distributed representation 으로 표현하며, 동일한 features (contexts) 로 기술되는 unseen entities 에 대한 distributed representation 을 추정하기 위한 작업입니다.


### Usage

2016-10-20 의 뉴스 기사에 대한 실험입니다. sentences 는 3 만여건의 뉴스기사 입니다. Input 은 list of str (like) 형식이면 모두 이용가능합니다.

```python
from text_embedding import Word2VecCorpus
from text_embedding import Word2Vec

corpus_path = '2016-10-20-news_noun'
word2vec_corpus = Word2VecCorpus(corpus_path)
word2vec = Word2Vec(word2vec_corpus)
```

학습된 Word2Vec 모델에서 Cosine distance 기준으로 '뉴스' 와 비슷한 단어를 찾습니다.

```python
word2vec.similar_words('뉴스')
```

    [('저작권자', 0.7723657073990423),
     ('기사', 0.7668060454902694),
     ('재배포', 0.7314359676352564),
     ('공감', 0.7235818366145983),
     ('금지', 0.7150778814892842),
     ('무단', 0.7115565530481157),
     ('뉴시스', 0.6645712933938247),
     ('전자신문', 0.621077137745182),
     ('영상', 0.6161913128674865),
     ('한국경제', 0.5922918560576881)]

Word2Vec 모델을 학습하는데 이용하였던 context words 를 기준으로 새로운 문서 집합에 대하여 word - context cooccurrance matrix 를 만들 수 있습니다. Cooccurrance matrix 의 rows 에 해당하는 단어는 word_set 에 입력합니다.

```python
additional_corpus = Word2VecCorpus(new_corpus_path)

WW, idx_to_vocab_ = word2vec.vectorize_word_context_matrix(
    additional_corpus,
    word_set = {'아이오아이', '너무너무너무'})
```

idx_to_vocab_ 은 새롭게 만들어진 word - context cooccurrance matrix 의 rows 에 해당하는 단어가 포함되어 있습니다.

```python
print(idx_to_vocab_)
# ['너무너무너무', '아이오아이']
```

WW 는 scipy.sparse.csr_matrix 입니다. WW 의 column 에 해당하는 단어는 아래에 저장되어 있습니다.

```python
word2vec._idx_to_vocab
```

Word - context cooccurrnace matrix 를 입력하면 새로운 단어에 대한 Word2Vec inference vector 를 얻을 수 있습니다.

```python
y = word2vec.infer_wordvec_from_vector(WW, idx_to_vocab_)
```

단어 벡터를 이용하여 유사어 검색이 가능합니다.

```python
word2vec.similar_words_from_vector(y[1].reshape(1,-1))
```

    [('아이오아이', 0.9990096169250022),
     ('완전체', 0.9188383418018635),
     ('엠카운트다운', 0.8988674553133801),
     ('엠넷', 0.8944098000201239),
     ('타이틀곡', 0.8893240912042681),
     ('너무너무너무', 0.8874956375432596),
     ('멤버들', 0.8733448469851375),
     ('오블리스', 0.8610452204993291),
     ('박진영', 0.8593707053005574),
     ('세븐', 0.8585589634747168)]


### References
- [1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). [Distributed representations of words and phrases and their compositionality][word2vec]. In Advances in neural information processing systems (pp. 3111-3119).
- [2] Levy, O., & Goldberg, Y. (2014). [Neural word embedding as implicit matrix factorization][wordpmi]. In Advances in neural information processing systems (pp. 2177-2185).

[word2vec]: http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases
[wordpmi]: http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization
