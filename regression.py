import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data") # 훈련 데이터셋 파일을 다운로드합니다. 경로를 반환한다


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True) #빈 행은 ?로 comment는 주석을 나타내는 기호를 알려줌 , space로 열을 구분

dataset = raw_dataset.copy()
dataset.tail() #마지막 5줄(기본값)을 출력한다
#판다스를 이용해 데이터를 읽음 

dataset = dataset.dropna() #누락된 항 삭제

origin = dataset.pop("Origin") #origin은 수치형이 아니므로 원-핫 인코딩 변환을 위해 빼줌

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail() 

train_dataset = dataset.sample(frac=0.8,random_state=0) #랸덤한 무작위 샘츨 추철 ,전체 개수에서 0.8개 반환, random_state는 춫ㄹ시 시드를 입력받고 같은시드에서는 같은결과 출력
test_dataset = dataset.drop(train_dataset.index) #인덱스 드랍

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde") #산점도 행렬을 만듦

train_stats = train_dataset.describe() #통계로 확인
train_stats.pop("MPG")
train_stats = train_stats.transpose()
  
train_labels = train_dataset.pop('MPG')#레이블 분리함
test_labels = test_dataset.pop('MPG')

def norm(x): #데이터 정규화 시킴
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
#훈련과정
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

class PrintDot(keras.callbacks.Callback): #에포크가 끝날때마다 (.)을 출력해서 진행괒멍 표시함
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()  

hist.savefig('mytable.png')