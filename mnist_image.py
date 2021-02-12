import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)), #2차원 배열인 이미지 포맷을 1차원배열로 변환 (이미지에 있는 픽셀의 행을 펼쳐서 일렬로 늘림) - 학습되는 가중치는 없고 그냥 늘리기만 한다
  tf.keras.layers.Dense(128, activation = 'relu'), #밀집연결 or 완전연결층 , 128개의 노드를 가짐
  tf.keras.layers.Dropout(0.2), #신경망연결 에서 일부를 끊어서(덜학습 시켜서) overfitting 방지
  tf.keras.layers.Dense(10, activation = 'softmax') # 10개의 노드의 소프트맥스층, 10개의 확률을 반환한다. 각 노드는 10개의 클래스(결과겂)에 속할 확률을 보여준다
])

model.compile(optimizer = 'adam',
loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5) #전체데이터를 5번에 걸쳐서 학습함 epoch 값이 너무 작다면 underfitting이(머신러닝 모델이 새로운 데이터에서 성능이 높아짐) 너무 크다면 overfitting이(머신러닝 모델이 새로운 데이터에서 성능이 낮아짐) 발생할 확률이 높다
model.evaluate(x_test, y_test, verbose = 2) #verbose = 0, 어떤  과정도 안보여줌; verbose = 1 프로그래스 바 보여줌, verbose = 2 epoch의 진생상황 알려줌

predictions = model.predict(x_test)

######여기부턴 그래프########


