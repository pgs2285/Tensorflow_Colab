import os
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer = 'adam',
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = y_train[:1000]
y_test = y_test[:1000]

x_train = x_train[:1000].reshape(-1, 28*28) / 255.0
x_test = x_test[:1000].reshape(-1, 28*28) / 255.0 #행에 -1이 들어갔으면 28*28의 리스트로 묶어준다는 뜻 !열에있으면 전체에서 행에있는 숫자로 나눈 리스트개수를 가짐


model = create_model()

model.summary()
                                                                                                                               

# 훈련중 체크포인트 저장하기

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) #입력받은 파일 디렉토리 경로 출력
                                                                                                                            
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, #체크포인트(checkpoint)를 자동으로 저장하도록 하는 것이 많이 사용하는 방법, 다시 훈련하지 않고 모델을 재사용하거나 훈련 과정이 중지된 경우 이어서 훈련을 진행
    save_weights_only=True,
    verbose=1
)                                                                                                                            

model.fit(x_train,
    y_train,
    epochs = 10,
    validation_data = (x_test, y_test),
    callbacks = [cp_callback] ## 콜백을 훈련에 전달
)

#텐서플로 체크포인트 파일을 만들고 에포크 종료마다 업데이트함

model = create_model() #테스트를 위해 이전모델이 아닌 모델 다시생성

loss, acc = model.evaluate(x_test, y_test, verbose = 2 )
print("훈련되지 않은 모델의 정확도 {:5.2f}%".format(acc*100)) #위에 fit한게 아닌 그냥 평가함

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(x_test, y_test, verbose = 2 ) #재평가
print("복원된 모델의 정확도 {:5.2f}%".format(acc*100))
