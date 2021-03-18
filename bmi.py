import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

csv = pd.read_csv("bmi.csv") #읽어들이고 정규화
csv["weight"] /= 100 # 몸무게는 40~100까지만 나옴
csv["height"] /= 200 # 키는 120~200까지만 나옴

X = csv[["weight", "height"]].values # Pandas로 CSV 파일을 읽어 들이고 원하는 열을 ndarray 자료형으로 변환

bclass = {"저체중":[1,0,0], "정상체중":[0,1,0], "비만":[0,0,1]} #레이블
y = np.empty((20000, 3)) #20000,3 배열 생성

for i, v in enumerate(csv["label"]):
    y[i] = bclass[v]

#테스트와 훈련데이터로 나누기
X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test = X[15001:20001], y[15001:20001]

# model = keras.Sequential()
# model.add(keras.layers.Dense(512, input_shape(2,)))
# model.add(activation = 'relu')
# model.add(Dropout(0.1))
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(2,)), #2차원 배열인 이미지 포맷을 1차원배열로 변환 (이미지에 있는 픽셀의 행을 펼쳐서 일렬로 늘림) - 학습되는 가중치는 없고 그냥 늘리기만 한다
    keras.layers.Dense(512, activation = 'relu'), #밀집연결 or 완전연결층 , 128개의 노드를 가짐
    keras.layers.Dropout(0.1), #신경망연결 에서 일부를 끊어서(덜학습 시켜서) overfitting 방지
    keras.layers.Dense(3, activation = 'softmax') # 10개의 노드의 소프트맥스층, 10개의 확률을 반환한다. 각 노드는 10개의 클래스(결과겂)에 속할 확률을 보여준다
])

# model.add(keras.layers.Dense(512))
# model.add(activation = 'relu')
# model.add(Dropout(0.1))

# model.add(keras.layers.Dense(3))
# model.add(activation = 'softmax')

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

hist = model.fit(
    X_train,y_train,
    batch_size=100,
    epochs=20,
    validation_split=0.1,
    verbose=1
)

score = model.evaluate(X_test, y_test)
print("loss = ", score[0])
print("accuracy = ", score[1])