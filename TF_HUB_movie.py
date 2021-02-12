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

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))# batch 는 주어진 크기로 데이터 세트를 자동으로 처리