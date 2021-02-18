import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("버전:", tf.__version__)
print("즉시 실행 모드:", tf.executing_eagerly())
print("허브 버전:", hub.__version__)
print("GPU:", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가")
# 훈련 세트를 6대 4로 나눕니다.
# 결국 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))# batch 는 주어진 크기로 데이터 세트를 자동으로 처리 , iter은 두번째 인자가 나올때까지 반복 , next는 다음것 가져요기

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" #사전 훈련된 텍스트 임베딩 모델
hub_layer = hub.KerasLayer(embedding, input_shape = [], dtype = tf.string, trainable = True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits= True), metrics = ['accuracy']) #결과가 0또는 1인 이진으로 나오므로 binarycrossentropy사용

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)


