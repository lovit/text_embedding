## Word vector inference

Word2Vec 은 words 를 크기가 정해진 공간 안의 distributed representation 으로 표현합니다. Skip-gram with Negative Sampling (SGNS) [1] 는 Word2Vec 의 구조 중 하나로, Softmax 의 효율적인 학습을 위하여 negative samplings 을 이용합니다.

Levy and Goldberg (2014) 는 [2] 에서 SGNS 는 (word, context) matrix 에 Shifted Positive Point Mutual Information (SPPMI) 를 적용한 결과임을 수학적, 실험적으로 증명하였습니다. PMI 는 informative contexts 를 선택하는 역할을 하며, Singular Value Decomposition (SVD) 는 유용한 문맥만으로 표현된 (word, context) matrix 를 정해진 크기의 공간으로 축소합니다.

학습데이터에 대하여 PMI 를 계산할 때 이용하는 context words probability 와 SVD 의 components 를 기억한다면, 새로운 단어에 대해서도 (word, context) matrix 를 만든 뒤, 동일한 PMI, SVD 를 적용하여, 학습데이터와 동일한 semantic space 의 word vector 로 새로운 단어들의 distributed representation 을 추정할 수 있습니다.

이 프로젝트는 PMI 와 SVD 를 이용하여 word (n-gram, phrase, doc and other entities) 를 distributed representation 으로 표현하며, 동일한 features (contexts) 로 기술되는 unseen entities 에 대한 distributed representation 을 추정하기 위한 작업입니다.


### Usage

2016-10-20 의 뉴스 기사에 대한 실험입니다. sentences 는 3 만여건의 뉴스기사 입니다. Input 은 list of str (like) 형식이면 모두 이용가능합니다.

    import soynlp
    from soynlp.utils import DoublespaceLineCorpus

    sentences = DoublespaceLineCorpus(corpus_path, iter_sent=True)

단어, '아이오아이'를 지운 뒤, 이를 inference 하기 위해 tokenizer 를 다르게 설정합니다. Default tokenizer 는 띄어쓰기 기준으로 단어를 나눕니다. 임의의 tokenizer 를 이용할 수 있습니다.

    def my_tokenizer(sent, passwords={'아이오아이'}):
        words = [word for word in sent.split() if not (word in passwords)]
        return words

wordvec_infer 에서 Word2Vec 을 import 한 뒤, 앞서 정의한 '아이오아이'를 제외하는 tokenizer 를 이용합니다.

    from wordvec_infer import Word2Vec

    word2vec = Word2Vec(tokenizer=my_tokenizer)
    word2vec.train(sentences)

학습된 Word2Vec 모델에서 Cosine distance 기준으로 '너무너무너무'와 비슷한 단어를 찾습니다. API 는 gensim 과 동일합니다.

    word2vec.most_similar('너무너무너무')
    
    [('신용재', 0.9270464973913624),
     ('완전체', 0.914970384426004),
     ('타이틀곡', 0.905864914102968),
     ('엠카운트다운', 0.9041150133215388),
     ('백퍼센트', 0.9014783849405605),
     ('몬스타엑스', 0.9010203753612409),
     ('곡으로', 0.8990371268486782),
     ('안무', 0.8907120459528796),
     ('박진영', 0.8871723098381121),
     ('신곡', 0.8833824952633795)]

하지만 단어 '아이오아이'는 토크나이저 과정에서 제외되어서 word vector 가 학습되지 않았습니다.

    word2vec.most_similar('아이오아이')
    
    []

원하는 단어들의 벡터를 추정합니다. infer 함수에 다음의 arguments 를 입력합니다.

| Variable name | Type | Description | 
| --- | --- | --- |
| sentences | list of str (like) | Training sentence |
| words | set of str (like) | Words we want to infer their vector |
| append | Boolean | Store inference results to Word2Vec model if it is True. Default is True |
| tokenizer | functional | tokenizer function. Default is lambda x:x.split() |

Word2Vec.infer 함수는 words 의 word vector 와 index 를 return 합니다. append=True 로 설정하면 이 결과가 모두 Word2Vec 모델에 저장됩니다.

    wordvec, index = word2vec.infer(
        sentences,
        words={'아이오아이'},
        append=True,
        tokenizer=lambda x:x.split()
    )

추정이 끝난 모델에 다시 한 번 단어 '너무너무너무'를 입력합니다. '아이오아이'가 유사 단어로 선택됩니다.

    word2vec.most_similar('너무너무너무')
    
    [('신용재', 0.9270464973913624),
     ('아이오아이', 0.9162263412577677),
     ('완전체', 0.914970384426004),
     ('타이틀곡', 0.905864914102968),
     ('엠카운트다운', 0.9041150133215388),
     ('백퍼센트', 0.9014783849405605),
     ('몬스타엑스', 0.9010203753612409),
     ('곡으로', 0.8990371268486782),
     ('안무', 0.8907120459528796),
     ('박진영', 0.8871723098381121)]

단어 '아이오아이'의 유사어도 검색합니다. 유사 단어들이 검색됩니다.

    word2vec.most_similar('아이오아이')

    [('엠카운트다운', 0.9243012341336443),
     ('엠넷', 0.9219115581331467),
     ('완전체', 0.91625257534599),
     ('너무너무너무', 0.9162263412577677),
     ('타이틀곡', 0.9074516014443481),
     ('몬스타엑스', 0.9061148638752767),
     ('멤버들', 0.9013150455703564),
     ('오블리스', 0.9005074700480684),
     ('신용재', 0.8961139817184961),
     ('백퍼센트', 0.8934708002132166)]

### References
- [1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). [Distributed representations of words and phrases and their compositionality][word2vec]. In Advances in neural information processing systems (pp. 3111-3119).
- [2] Levy, O., & Goldberg, Y. (2014). [Neural word embedding as implicit matrix factorization][wordpmi]. In Advances in neural information processing systems (pp. 2177-2185).

[word2vec]: http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases
[wordpmi]: http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization