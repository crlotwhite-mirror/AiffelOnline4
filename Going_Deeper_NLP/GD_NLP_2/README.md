## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 소용현

## **PRT(PeerReviewTemplate)**
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
머신러닝 모델 및 딥러닝 모델 모두 80%이상의 정확도를 달성하는 등 주어진 문제를 해결하였다.
- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
```
import matplotlib.pyplot as plt

# 서브플롯 생성
# 2행 2열의 구조로 총 4개의 그래프를 생성한다.
fig, axs = plt.subplots(2, 2, figsize=(15,10)) 

# 2차원 numpy array를 1차원으로 변환한다. 
# 이를 통해 각 그래프에 순차적으로 접근할 수 있다.
axs = axs.ravel() 

# 출력할 평가 지표들을 리스트로 정의한다.
measures = ['acc', 'pre', 'rec', 'f1']

# 각 평가 지표에 대해 반복문을 수행한다.
for i, measure in enumerate(measures):
    # common_dict 내의 각 모델에 대한 결과 값들을 순회한다.
    for model, model_result in common_dict.items():
        # 각 모델의 결과에서 단어 수를 추출한다. (key로 존재)
        num_words = list(model_result.keys())  
        # 해당하는 평가 지표의 값을 추출한다.
        scores = [model_result[n][measure] for n in num_words]  
        # 각 모델의 단어 수에 따른 평가 지표를 그래프로 그린다.
        axs[i].plot(num_words, scores, marker='o', label=model)  
    # 그래프의 제목을 설정한다. (대문자로 표시)
    axs[i].set_title(measure.upper())
    # x축의 라벨을 설정한다.
    axs[i].set_xlabel('Number of Words')
    # y축의 라벨을 설정한다.
    axs[i].set_ylabel('Score')
    # 범례를 설정한다.
    axs[i].legend()
    # 그리드를 표시한다.
    axs[i].grid()

# 각 그래프의 위치가 겹치지 않도록 조정한다.
plt.tight_layout()
# 그래프를 화면에 출력한다.
plt.show()
```
상세한 주석으로 시각화를 표현하였다.
- [O] **3. 코드가 에러를 유발할 가능성이 있나요?**
```
    # 인덱스를 단어로 매핑하는 딕셔너리를 만듭니다.
    index_to_word = {index+3: word for word, index in word_index.items()}
    index_to_word[0] = "<PAD>"
    index_to_word[1] = "<START>"
    index_to_word[2] = "<UNKNOWN>"  # unknown
    index_to_word[3] = "<UNUSED>"
```
로이터 데이터셋을 로딩하고 시퀀스와 매핑하는 과정에서, 0,1,2번 매핑을 로이터 데이터 셋에 맞추어야 에러 가능성을 없앨 수 있을 것 같다.
노드에 따르면 0번은 <pad>, 1번은 <sos>, 2번은 <unk>로 나타나 있다.  

- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
```
from concurrent.futures import ThreadPoolExecutor, as_completed

def train_and_evaluate_model(model, kwargs, dataset):
    print(f'{model.__name__}-case{dataset["num_words"]} start') # 시작 로그

    # 모델 생성 및 학습, 그리고 예측
    m = model(**kwargs)
    m.fit(dataset['train']['x'], dataset['train']['y'])
    pred = m.predict(dataset['test']['x'])

    # 측정
    precision, recall, fscore, _ = precision_recall_fscore_support(dataset['test']['y'], pred, average='macro', warn_for=tuple())
    acc = accuracy_score(dataset['test']['y'], pred)
    print(f'{model.__name__}-case{dataset["num_words"]} done') # 종료 로그
    return model.__name__, dataset["num_words"], acc, precision, recall, fscore

common_dict = {}
with ThreadPoolExecutor(max_workers=24) as executor:
    # 작업 큐
    futures = []
    for model, kwargs in models:
        # 딕셔너리에 모델을 위한 공간 할당
        common_dict[model.__name__] = {k: {'acc': {}, 'pre': {}, 'rec': {}, 'f1': {}} for k in [5000, 10000, 15000]}
        
        # 비동기 작업 큐잉
        for dataset in datasets:
            futures.append(executor.submit(train_and_evaluate_model, model, kwargs, dataset))
    
    # 병렬 처리 시작
    for future in as_completed(futures):
        # 모든 처리가 끝나고 Dict에 업데이트
        # 공통 루틴에서 처리하므로서 레이스컨디션 및 동시성 이슈 방지
        model_name, case_no, acc, precision, recall, fscore = future.result()
        common_dict[model_name][case_no]['acc']=acc
        common_dict[model_name][case_no]['pre']=precision
        common_dict[model_name][case_no]['rec']=recall
        common_dict[model_name][case_no]['f1']=fscore
```
작업시간을 줄이기 위해 쓰레드를 사용하여 병렬처리하였다.
- [O] **5. 코드가 간결한가요?**
```
# 모델 리스트 생성

models = [
    (MultinomialNB, {}), # 나이브 베이즈 분류기
    (ComplementNB, {}), # 컴플리먼트 나이브 베이즈
    (LogisticRegression, {'C':10000, 'penalty': 'l2', 'max_iter': 3000}), # 로지스틱 회귀
    (LinearSVC, {'C':1000, 'penalty': 'l1', 'max_iter': 3000, 'dual': False}), # 서포트 벡터 머신
    (DecisionTreeClassifier, {'max_depth': 10, 'random_state': 0}), # 결정 트리
    (RandomForestClassifier, {'n_estimators': 5, 'random_state': 0}), # 랜덤 포레스트
    (GradientBoostingClassifier, {'random_state': 0}), # 그래디언트 부스팅
]

# 이미 있는 조건을 이용해서 voting 앙상블 구성
models.append(
    (VotingClassifier, 
        {
            'voting': 'soft', 
            'estimators': [(models[x][0].__name__, models[x][0](**models[x][1])) for x in [2, 1, 6]]
        }
    )
)
```
좋은 방법으로 코드를 간결화 하였다.

## **참고링크 및 코드 개선 여부**

    
