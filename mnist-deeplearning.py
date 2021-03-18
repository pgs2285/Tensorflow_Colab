import tensorflow as tf

mnist = tf.keras.datasets.mnist

def weight_variable(name, shape): # 가중치와 바이어스 초기화
    W_init = tf.random.truncated_normal(shape, stddev=0.1) #랜덤(정규분포에서)으로 수를준다
    W = tf.Variable(W_init, name ="W_"+ name)
    return W

def bias_variable(name, size):
    b_init = tf.constant(0.1, shape=[size])
    b = tf.Variable(b_init, name="b_"+name)
    return b

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') #합성곱 계층을 만든다

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1]) #최대 풀링층 만듬

with tf.name_scope('conv1') as scope: #합성곱층1
    W_conv1 = weight_variable('conv1', [5, 5, 1, 32])       
    b_conv1 = bias_variable('conv1', 32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope('pool1') as scope: #풀링층1
    h_pool1 = max_pool(h_conv1)

with tf.name_scope('conv2') as scope: #합성곱층2
    W_conv2 = weight_variable('conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable('conv2', 64)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

with tf.name_scope('pool2') as scope: #풀링층2
    W_conv2 = weight_variable('conv2', [5, 5, 32, 65])    

with tf.name_scope('fully_connected') as scope: #전결합층
    n = 7 * 7 * 64 
    W_fc = weight_variable('fc', [n, 1024])
    b_fc = bias_variable('fc', 1024)
    h_pool2_flat = tf.reshape(h_pools, [-1, n])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc)+b_fc)

with tf.name_scope('dropout') as scope:
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

with tf.name_scope('readout') as scope:
    W_fc2 = weight_variable('fc2', [1024, 10])
    b_fc2 = bias_variable('fc2', 10)
    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc) + b_fc2)

with tf.name_scope('loss') as scope:
    cross_entoropy = -tf.reduce_sum(y_ * tf.log(y_conv))

with tf.name_scope('training') as scope:
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entoropy)

with tf.name_scope('predict') as scope:
    predict_step = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))

    