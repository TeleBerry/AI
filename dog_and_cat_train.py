import ReadMyOwnData  # 讀取自己建的 py檔
from keras.layers.convolutional import Conv2D
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


epoch = 15  # 訓練15期
batch_size = 20  # 資料集大小


def one_hot(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j]))
                               for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


def weight_variable(shape):  # initial weights
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)


def bias_variable(shape):  # initial bias
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
y_ = tf.placeholder(tf.float32, [batch_size, 2])


# sess = tf.InteractiveSession()
# with tf.name_scope('Input'):
#     x = tf.placeholder(tf.float32, shape=[None, n_features])
# with tf.name_scope('Label'):
#     y_ = tf.placeholder(tf.float32, shape=[None, n_labels])


def conv2d(x, W):  # convolution layer
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_4x4(x):  # max_pool layer
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


with tf.name_scope('FirstConvolutionLayer'):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_4x4(h_conv1)

with tf.name_scope('SecondConvolutionLayer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)

# 變全連接層，用一個 MLP 處理
reshape = tf.reshape(h_pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 損失函数及優化算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

img, label = ReadMyOwnData.read_and_decode("dog_and_cat_train.tfrecords")
img_test, label_test = ReadMyOwnData.read_and_decode(
    "dog_and_cat_test.tfrecords")

# 使用 shuffle_batch可以随機打亂输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                              batch_size=batch_size, capacity=2000,
                                              min_after_dequeue=1000)

init = tf.initialize_all_variables()
t_vars = tf.trainable_variables()
print(t_vars)

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_idxs = int(2314/batch_size)
    for i in range(epoch):
        for j in range(batch_idxs):
            val, l = sess.run([img_batch, label_batch])
            l = one_hot(l, 2)
            _, acc = sess.run([train_step, accuracy], feed_dict={
                              x: val, y_: l, keep_prob: 0.5})
            print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" %
                  (i, j, batch_idxs, acc))
    val, l = sess.run([img_test, label_test])
    l = one_hot(l, 2)
    print(l)
    y, acc = sess.run([y_conv, accuracy], feed_dict={
                      x: val, y_: l, keep_prob: 1})
    print(y)
    print("test accuracy: [%.8f]" % (acc))
    coord.request_stop()
    coord.join(threads)