## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 남희정

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  네네. 코드가 정상적으로 동작하고 주어진 문제를 잘 해결하였습니다.
* 한글 전처리 주요 코드
```python
 regex = r"[^a-zA-Z가-힣?.!,]+"
```
* 주어신 함수의 paramater 값이 잘 적용된 SentencePieceTrainer 생성
```python
def generate_tokenizer(corpus,
                        vocab_size,
                        lang="ko",
                        pad_id=0,
                        bos_id=1,
                        eos_id=2,
                        unk_id=3):
    temp_file = f'{lang}_temp.txt'
    model_prefix = f'{lang}_spm'
    
    # 주어진 코퍼스를 사용하여 임시 텍스트 파일 생성
    with open(temp_file, 'w', encoding='utf-8') as f:
        for sentence in corpus:
            f.write(sentence + '\n')
    
    # 임시 파일을 이용하여 SentencePiece 모델 학
    spm.SentencePieceTrainer.Train(
        f'--input={temp_file} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} --unk_id={unk_id}')
    
    # 학습된 모델을 불러와 SentencePieceProcessor 객체 생성
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f'{model_prefix}.model')
```
* 토큰 길이 50이상 제거
```python
for idx in tqdm(range(len(kor_corpus))):
    src_tokens = ko_tokenizer.EncodeAsIds(kor_corpus[idx])
    tgt_tokens = en_tokenizer.EncodeAsIds(eng_corpus[idx])
    
    # 한국어와 영어 모두가 조건에 부합하는 데이터만 이용함
    if len(src_tokens) <= 50 and len(tgt_tokens) <= 50:
        src_corpus.append(src_tokens)
        tgt_corpus.append(tgt_tokens)
```
* Tranformer에 필요한 단계별 기능들은 노드와 동일하므로 생략
* 마지막 실행단에 loss값 저장하여 시각화 사용
```python
import random

examples = [
            "오바마는 대통령이다.",
            "시민들은 도시 속에 산다.",
            "커피는 필요 없다.",
            "일곱 명의 사망자가 발생했다."
]

# 각 에폭에서의 평균 손실을 추적하기 위한 리스트
losses = []

for epoch in range(EPOCHS):
    # 총 loss
    total_loss = 0
    
    # 배치 크기 단위로 훈련 데이터 인덱스를 생성하고 무작위로 섞는다.
    idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
    random.shuffle(idx_list)
    t = tqdm(idx_list)
    
    # 미니배치 학습 진행
    for (batch, idx) in enumerate(t):
        # 학습 루틴 호출
        batch_loss, enc_attns, dec_attns, dec_enc_attns = \
        train_step(enc_train[idx:idx+BATCH_SIZE],
                    dec_train[idx:idx+BATCH_SIZE],
                    transformer,
                    optimizer)
        
        # 배치 손실 추가
        total_loss += batch_loss
        
        # tqdm 업데이트
        t.set_description_str('Epoch %2d' % (epoch + 1))
        t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
    
    # 최종 손실 추가
    losses.append(total_loss.numpy() / (batch + 1))
    
    # 해당 에포크에 대한 번역 결과 출력
    for example in examples:
        translate(example, transformer, ko_tokenizer, en_tokenizer)
    ```    
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
  네. 잘이해 되었습니다. 특히 loss 값을 시각화한 부분이 결과를 확인하고 이해하는데 도움이 되었습니다.   

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
  loss 값을 시각화 하는 부분은 저도 시도했던 부분이라 제 코드에 참고하도록 하겠습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
  네. 잘 이해하고 작성한것으로 판단됩니다.

- [x] **5. 코드가 간결한가요?**
  네. 코드는 간결하게 잘 작성되었습니다. 

## **참고링크 및 코드 개선 여부**

    
