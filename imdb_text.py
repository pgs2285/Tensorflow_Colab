import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb #컴퓨터에 imdb셋 다운 (이미 다운한적 있으면 캐시된 복사본 사용)
(x_train,y_train), (x_test, y_test) = imdb.load_data(num_words = 10000) #numwords = 10000 은 훈련 데이터에서 가장 많이 등장하는 10000개의 단어를 선택함
#y값에서 0은 부정적인 리뷰 1은 긍정적인 리뷰이다

word_index = imdb.get_word_index()#인트형 리스트로되어있는 x값을 바꿈
 
word_index = {k:(v+3) for k, v in word_index.items()}# 앞에 4개는 따로 정의할거라 v+3을 한다
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # key와 value의 위치를 바꿔준다

def decode_review(text):
    return " ".join([reverse_word_index.get(i) for i in text])
#숫자리스트 영어리뷰로 변환 

x_train = keras.preprocessing.sequence.pad_sequences(x_train,
value = word_index["<PAD>"], #선호값 지정?
padding ='post',
maxlen = 256) #패드 시퀀스 최대길이 지정 256보다 짧으면 0으로 채우고 길면 잘라낸다


x_test = keras.preprocessing.sequence.pad_sequences(x_test,
value = word_index["<PAD>"],
padding ='post',
maxlen = 256)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,))) #임베딩 벡터의 크기는 16 즉, 10000개의 단어에 대한 임베딩 벡터 10000개만듬
model.add(keras.layers.GlobalAveragePooling1D()) # 고정된 크기의 출력벡터를 리턴, 압력shape(25000,256,16)배열 사용, 두번째 차원 방향으로 평균을 구하여 (25000, 16)배열 생성
model.add(keras.layers.Dense(16, activation='relu'))#16개의 뉴런에 입력백터 출력 ,크기 16배열 리턴
model.add(keras.layers.Dense(1, activation='sigmoid'))# 하나의 노드로 구성된 출력 레이어에 fully connected
model.summary()

model.compile(optimizer = 'adam',
loss = 'binary_crossentropy',
metrics = ['accuracy'])

x_val = x_train[:10000] #테스트 이미지 사용 x  
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
partial_y_train,
epochs = 35,#에포크 40번 반복
batch_size = 512, #512개의 샘플로 이루어진 mini배치에
validation_data = (x_val, y_val),
verbose = 1)

results = model.evaluate(x_test, y_test ,verbose = 2)
print(results)
