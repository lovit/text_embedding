## Inference experiments

### type 1 vs type 2

Type 1 은 word - contexts co-occurrence matrix 에서 test words 를 삭제한 뒤, PMI, Py 를 계산하여 test words 에 대한 word vector inference 를 진행하는 것이며, 

Type 2 는 test words 가 포함되지 않은 문장들만을 이용하여 base Word2Vec model 을 학습한 뒤, test words 가 포함된 문장들만 이용하여 새롭게 word - context co-occurrence matrix 를 만들어 inference 를 진행한다.

Type 2 에서는 base model 을 학습할 때, 학습데이터 전체의 일부만을 이용하기 때문에 min_count 조건을 만족하는 vocabulary 의 개수가 줄어든다.

compare 함수에서는 이를 보정하기 위하여 full data model 과 small data model 에서 공통으로 등장한 단어만을 이용하여 similar word search 결과를 비교한다.
