## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**

```python
# 쉘 스크립트 실행
# 폴더 생성
!mkdir -p ~/aiffel/camera_sticker/models
!mkdir -p ~/aiffel/camera_sticker/images
# 심볼릭 링크 생성
!ln -s ~/data/* ~/aiffel/camera_sticker/images
# 모델 다운로드
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!mv shape_predictor_68_face_landmarks.dat.bz2 ~/aiffel/camera_sticker/models

# 압축 해제
!cd ~/aiffel/camera_sticker && bzip2 -d ./models/shape_predictor_68_face_landmarks.dat.bz2
# %cd를 쓰면 노트 자체의 디렉토리가 바뀜
```
와 같이 과정의 진행을 주석으로 설명해주셔서 이해가 수월했습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
```python
# 얼굴 감지기 호출
detector_hog = dlib.get_frontal_face_detector()
print('Done')
```
```python
# 탐지
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)  
print('Done')
```
호출, 선언, 결과 등을 확인하며 진행하여 에러 유발 가능성이 적어보입니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
```python
# 주요 교점을 유추하고
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    # 교점 좌표 1
    x1 = landmark[2][0]; y1 = landmark[2][1] # 코 옆 얼굴 점
    x2 = landmark[30][0]; y2 = landmark[30][1] # 콧등 아래쪽
    x3 = landmark[36][0]; y3 = landmark[36][1] # 왼쪽 눈 끝
    x4 = landmark[48][0]; y4 = landmark[48][1] # 왼쪽 입술 끝
    
    # 교점 좌표 2
    x5 = landmark[14][0]; y5 = landmark[2][1] # 코 옆 얼굴 점
    x6 = landmark[30][0]; y6 = landmark[30][1] # 콧등 아래쪽
    x7 = landmark[45][0]; y7 = landmark[36][1] # 오른쪽 눈
    x8 = landmark[54][0]; y8 = landmark[48][1] # 오른쪽 입술
    
    # 교점 좌표 1 계산
    Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))//((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))//((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    cv2.circle(img_show, (Px,Py), 5, (0, 0, 255), -1)
    
    # 교점 좌표 2 계산
    Px2 = ((x5*y6 - y5*x6)*(x7 - x8) - (x5 - x6)*(x7*y8 - y7*x8))//((x5 - x6)*(y7 - y8) - (y5 - y6)*(x7 - x8))
    Py2 = ((x5*y6 - y5*x6)*(y7 - y8) - (y5 - y6)*(x7*y8 - y7*x8))//((x5 - x6)*(y7 - y8) - (y5 - y6)*(x7 - x8))
    cv2.circle(img_show, (Px2,Py2), 5, (0, 0, 255), -1)
    
    # 스티커 위치를 위한 높이, 넓이 계산
    w = Px2 - Px
    h = Px2 - Px
    
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show)
    plt.show()
```
코드를 이해하고 스티커의 좌표를 설정하였습니다.

- [x] **5. 코드가 간결한가요?**

```python
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x1 = landmark[2][0]; y1 = landmark[2][1]
        x2 = landmark[30][0]; y2 = landmark[30][1]
        x3 = landmark[36][0]; y3 = landmark[36][1]
        x4 = landmark[48][0]; y4 = landmark[48][1]

        x5 = landmark[14][0]; y5 = landmark[2][1]
        x6 = landmark[30][0]; y6 = landmark[30][1]
        x7 = landmark[45][0]; y7 = landmark[36][1]
        x8 = landmark[54][0]; y8 = landmark[48][1]

        Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))//((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
        Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))//((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
        cv2.circle(img_show, (Px,Py), 5, (0, 0, 255), -1)


        Px2 = ((x5*y6 - y5*x6)*(x7 - x8) - (x5 - x6)*(x7*y8 - y7*x8))//((x5 - x6)*(y7 - y8) - (y5 - y6)*(x7 - x8))
        Py2 = ((x5*y6 - y5*x6)*(y7 - y8) - (y5 - y6)*(x7*y8 - y7*x8))//((x5 - x6)*(y7 - y8) - (y5 - y6)*(x7 - x8))
        cv2.circle(img_show, (Px2,Py2), 5, (0, 0, 255), -1)


        w = Px2 - Px
        h = Px2 - Px
```
상대적으로 길었던 부분이었음에도 변수 선언, 계산 수식 부분이 적절하게 나누어져 있어 가독성이 매우 좋았습니다!



    
