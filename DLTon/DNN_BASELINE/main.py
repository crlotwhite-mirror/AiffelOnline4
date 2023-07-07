import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Mecab
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

# 데이터 불러오기
data = pd.read_csv('data.csv')

# class 열을 숫자로 변환
encoder = LabelEncoder()
data['Class'] = encoder.fit_transform(data['Class'])

# konlpy의 Mecab을 이용한 토큰화
mecab = Mecab()
data['Conversation'] = data['Conversation'].apply(mecab.morphs)

# 케라스를 이용한 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Conversation'])

sequences = tokenizer.texts_to_sequences(data['Conversation'])

# 입력 데이터 패딩 처리
data_pad = pad_sequences(sequences)

# 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(data_pad, data['Class'], test_size=0.1, random_state=42)

# MLP 모델 생성
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=data_pad.shape[1]))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(data['Class'].unique()), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

test_x = '''우리팀에서 다른팀으로 갈 사람 없나? 그럼 영지씨가 가는건 어때? 
네? 제가요? 
그렇지? 2달만 파견 잘 갔다오면 승진이야. 
네? 저는 별로 가고 싶지 않습니다. 
여기 있는 모든사람도 가기 싫어해. 그러니까 막내인 영지씨가 가는게 맞지 
정말 죄송합니다. 저는 못갑니다. 
장난해? 모두를 위해 영지씨가 희생하는게 싫어? 
네. 부당한 방법으로 가는 것 같습니다. 
영지씨 안가면 회사생활 오래 못할 것 같은데 그래도 안갈거야? 안가면 지옥일텐데. 
그래도 이 방법은 아닌 것 같습니다. 죄송합니다.'''

test_y = encoder.transform("직장 내 괴롭힘 대화")