## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 김재환

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
    - 코드 정상 작동, 모델을 통해 Abstract Summarizaiton 과 Extract Summarization을 이상없이 구현하였습니다.  
    - 특히, 데이터셋의 분포를 확인하여 `src_vocab`과 `tar_vocab` size를 적절하게 수정하였습니다.
    - 결과로, abstract summarization의 성능이 (리뷰어의 결과와 비교했을 때) 더 좋았습니다.

    - Abstract & Extract Summarization의 효과를 비교하기 위해 여러 개의 Case를 출력하였고,  
    - 다수의 결과를 종합해 `문장 길이`, `정확도` 등 다양한 지표로 분석한 결과를 작성하였습니다.  

- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
    - 큰 단위의 Task와 작은 단위의 Task를 구분하여 markdown headline을 지정하였습니다.  
    - 작은 단위의 Task의 경우, 코드라인에서의 주석을 통해 충분히 코드 기능을 확인할 수 있었습니다.

#### 잘 이해되었던 주석 예시
```python
# 단어 시퀀스를 형성하는 함수 생성
def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)

     # <SOS>에 해당하는 토큰 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tar_index_to_word[sampled_token_index]

        if (sampled_token!='eostoken'):
            decoded_sentence += ' '+sampled_token

        #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (headlines_max_len-1)):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence
print('Done')
```

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
    - 아니오. 에러 가능성이 보이지 않습니다.
    - 단계 별, 변수 별 주석 처리가 잘 되어있어, 입력 데이터, 경로 등의 변수로 에러가 발생하더라도  
    빠르게 해결이 가능할 것으로 보입니다.
    
    
- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
    - 1. 에서도 기술했듯, vocabulary 변수 값을 근거에 따라 적절하게 변경한 걸 확인했습니다.

- [x] **5. 코드가 간결한가요?**
    - 간결합니다! 한 코드블럭의 길이도 길지 않고, 한 줄의 길이가 길지 않아 가독성이 뛰어납니다. 

## **참고링크 및 코드 개선 여부**

    
