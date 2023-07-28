## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 김경훈

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

> * 기존 데이터셋의 문제점을 분석하고 전처리 전략을 수립해 추가 정제를 진행했습니다.

``` python
from koeda import EDA
import numpy as np

def run_eda(text):
    return eda(text, p=(0.9, 0.9, 0.9, 0.9), repetition=1)

eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.3, alpha_ri=0.0, alpha_rs=0.0, prob_rd=0.3
)

for i in [4, 5, 6]:
    df_cluster3 = df[df['cluster_label'] == i]

    # 랜덤하게 행 선택 (예: 선택된 행의 30%를 선택)
    random_indices = \
        np.random.choice(df_cluster3.index, size=int(len(df_cluster3)*0.3), replace=False)

    # 선택된 행에 대해 Random swap 함수 적용
    augmented_rows = df.loc[random_indices, 'prompt'].apply(run_eda)
    # 증강된 데이터를 복사하고, 'text' 열에 증강된 텍스트를 삽입
    new_rows = df.loc[random_indices].copy()
    new_rows['prompt'] = augmented_rows

    # 원본 데이터프레임에 증강된 데이터 추가
    df = pd.concat([df, new_rows])
```

> * 생성된 텍스트를 평가하기 위한 메트릭(BLEU)을 적용한 정량적인 평가 결과와 주관적인 평가를 비교분석하였습니다.

``` python
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric

# NLTK BLEU calculation function
def calculate_bleu(reference_sentence, generated_sentence):
    return sentence_bleu([reference_sentence], generated_sentence)

# Load the ROUGE metric
rouge_metric = load_metric("rouge")

def generation(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = actor.generate(input_ids,
                             max_length=250,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95,
                             num_return_sequences=1)
    generated_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs[0]]

    # Prepare reference sentences for BLEU and ROUGE
    reference_sentence = PROMPT_DICT['prompt_input'].format_map({'prompt': input_text})
    reference_sentences = [reference_sentence for _ in list_prompt]

    # BLEU and ROUGE evaluation
    bleu_scores = [calculate_bleu(reference, output.split()) for reference, output in zip(reference_sentences, generated_responses)]
    rouge_scores = rouge_metric.compute(predictions=generated_responses, references=[reference_sentences])

    print("Input Prompt:", input_text)
    print("Generated Response:", generated_responses[0])
    print("BLEU Scores:", bleu_scores)
    print("ROUGE Scores:", rouge_scores)
    print("="*50)

    return generated_responses

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    )
}

list_prompt = [
    '불고기용 고기 한우에요?', 
    '리처드 닉슨이 43대 부통령직을 수행한 년도는?', 
    '시카고 오헤어 국제공항은 어디에 있어',
    '오늘 미세먼지 어때?']

list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt': tmp}) for tmp in list_prompt]

for input_text in list_prompt:
    output = generation(input_text)
```
 
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**

> * 각 과정마다 테스크에 대한 설명의 주석이 작성되어 있습니다.

``` python
# 토크나이저 테스트
input_txt = "바람도 없는 공중에 수직의 파문을 내이며 고요히 떨어지는 오동잎은 누구의 발자취 입니까."

tokens = tokenizer(input_txt).tokens()
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].numpy()

# 디코더 확인
max_length=128
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output_greedy[0]))

# 빔서치 디코딩에 n-gram 패널티 부과
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output_beam = model.generate(input_ids, max_length=max_length, num_beams=10, no_repeat_ngram_size=2, do_sample=False)
print(tokenizer.decode(output_beam[0]))

# 샘플링 기법 추가 - top_k
output_beam = model.generate(input_ids, max_length=max_length, num_beams=7, no_repeat_ngram_size=2,
                             do_sample=True, temperature=2.0, top_k=50)
print(tokenizer.decode(output_beam[0]))
```

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**

에러를 유발할 가능성은 적어 보입니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**

코드를 제대로 이해하고 작성하였습니다.

- [x] **5. 코드가 간결한가요?**

코드가 간결합니다.

## **참고링크 및 코드 개선 여부**

    
