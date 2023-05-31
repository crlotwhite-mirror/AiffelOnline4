## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 사재원

## **PRT(PeerReviewTemplate)**
------------------  
- [&#11093;] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**  
  네, 코드는 정상적으로 동작하고 이번 프로젝트의 요구사항들 모두 충족시켰습니다. 
  loss와 accuracy의 시각화를 통해 모델 학습의 과정을 잘 보여준 것 같습니다. 

- [&#11093;] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  네 오히려 주석시킨 내용으로 인해 전체적인 흐름을 차근차근 되짚어 보는 느낌이었습니다. 덕분에 이번 프로젝트 내용에 대해 좀 더 정리가 되었습니다.
- [&#10060;] **3. 코드가 에러를 유발할 가능성이 있나요?**  
  아니요. 순차적으로 코드를 실행시켜본다면 따로 에러를 유발할 내용은 포함되어있지 않는 것 같습니다.
- [&#11093;] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  네 코드를 작성하기 이전 과정을 먼저 주석이나 따로 노트에 서술한 후 순차적으로 기능들을 구현하는 내용을 볼 수 있습니다.
  최적의 학습결과를 도출하기위해 batchsize와 같은 모델 파라미터를 수정한 내용도 확인할 수 있었습니다. 
   
   ```python
   # 위 과정을 담은 코드
    def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)

    # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
    # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
    sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
    # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
    output_sequence = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 인퍼런스 단계
    for i in range(MAX_LENGTH):
        # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

        # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
              break

        # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
        # 이 output_sequence는 다시 디코더의 입력이 됩니다.
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)

   ```
- [&#11093;] **5. 코드가 간결한가요?**  
  네 적절한 주석을 통하여 작성자가 어떤 의도를 가지고 코드를 작성하였는지 쉽게 이해할 수 있었습니다.
## **참고링크 및 코드 개선 여부**

    
