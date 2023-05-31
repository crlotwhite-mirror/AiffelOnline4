## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 

## **PRT(PeerReviewTemplate)**
평가문항	상세기준
1. pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다.	데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.
2. pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다.	U-Net generator, discriminator 모델 구현이 완료되어 train_step의 output을 확인하고 개선하였다.
3. 학습 과정 및 테스트에 대한 시각화 결과를 제출하였다.	10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.

------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- tf2는 Eager execution가 기본 실행인데, Eager execution에서 성능 이슈가 발생하여 이를 해결하기위한 Jitter의 개념을 적용하였다는점이 흥미로웠습니다. 단순히 Augmentaion의 기능을 이것저것 해보는것이 아니라, Augmentation이 필요한 이유를 설명하고 있다는 점이 좋았습니다.
- 최종적으로 resizing-> crop -> 좌우반전 순의 augmentation이 수행되었습니다. 
- <code>def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image</code>
- <code>def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
    stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]</code>
 - plot_model mothod를 사용해 input과 output의 dimension이 어떻게 바뀌어 가는지를 확인할 수 있는 부분으 특히 좋았고 저도 제코드에 적용해 보겠습니다. 
 - <code>generator = Generator()
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)</code>
    
    
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
- 단계별 필요한 설명을 주석으로 적절히 달아두었고, 특히 본인이 이해한 방식과 로직의 전체적인 흐름을 이해할 수 있어서 좋았습니다. 

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
- 코드는 제대로 이해하고 있다고 판단했습니다. 제코드를 Review 하시면서 하신 질문과 대화를 통해 그렇게 판다하였습니다. 

- [x] **5. 코드가 간결한가요?**
- [ ] U-net Generator 코드작성에서 노드에서 사용한 Class 형식이 아닌 함수형태의 Genterator를 만들었다는것이 흥미로운 부분이고 Class를 사용하여 만든것보다 보기에도 간결하고, 로직도 쉽게 이해가 되어서 더 나은 코드로 느껴지긴 했습니다만..
- <code>def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        # 잘 보면 다운 샘플러이지만 채널수가 증가한다.
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    up_stack = [
        # 여기는 점점 줄어드는 상황
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)    ]</code>
## **참고링크 및 코드 개선 여부**

    
