## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 남희정

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [ ] ![image](https://github.com/crlotwhite-mirror/AiffelOnline4/assets/88833290/abed3be0-5ad9-4772-b9dc-40ddc78ff6b3)
    ```python
    # 128000건만 메모리에 로딩
    pre_train_inputs, pre_train_labels = load_pre_train_data(vocab, pretrain_json_path, 128, count=128000)
    ```
   128000건만 test하여 test의 결과가 전체를 다 test 한것과는 상이하나, Data의 크기에 따른 issue이므로 전반적으로 주어진 문제들을 잘 해결하였습니다.

- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
    ```python
    instances = []
    current_chunk = []  # line 단위 tokens
    current_length = 0
    for i in range(len(doc)):  # doc 전체를 loop
        current_chunk.append(doc[i])  # line 단위로 추가
        current_length += len(doc[i])  # current_chunk의 token 수
    if 1 < len(current_chunk) and (i == len(doc) - 1 or current_length >= max_seq): 
    ```
    라인별 설명을 잘 작성하였습니다. 
- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
      글쎄요. 분석할 수 있는 능력을 키워보도록 하겠습니다. 

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
```python
            import sentencepiece as spm
            import os
            
            corpus_file = os.getenv('HOME')+'/aiffel/bert_pretrain/data/kowiki.txt'
            prefix = 'ko_8000'
            vocab_size = 8000 + 7 # 특수 문자 추가
```
```python
        if not os.path.exists(f'{prefix}.model'):
            spm.SentencePieceTrainer.train(
                f"--input={corpus_file} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
                " --model_type=bpe" +
                " --max_sentence_length=999999" + # 문장 최대 길이
                " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
                " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
                " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
                " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
                " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰
```
      네. 잘 이해하고 작성하였습니다. 실제로 sentencepiece model를 생성하여 사용하였습니다. 
- [x] **5. 코드가 간결한가요?**
      네. 간결하게 작성되었습니다. 

## **참고링크 및 코드 개선 여부**

    
