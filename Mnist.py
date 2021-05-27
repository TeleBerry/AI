# import tensorflow as tf
from tensorflow import input_data  # 找到我放在tensorflow的input_data

import tensorflow.compat.v1 as tf  # 匯入tensorflow
tf.disable_v2_behavior()

# 找到我的mnist資料集
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


# 定義神經層函數----------------------------------------------------------
def add_layer(inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer'):

        with tf.name_scope('weights'):  # 名稱為weights的隨機變量矩陣
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# ------------------------------------------------------------------


# 計算正確率-----------------------------------------------------------
def compute_accuracy(v_xs, v_ys):  # v_xs 輸入的mnist影像集、 v_ys 對應的label

    global prediction

   # 計算出的預測結果 y_pre與v_ys做對比，
   # 如果相同則判斷正確，否則為錯誤，計算出的正確結果儲存在correct_prediction 中

    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    corrct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    accuracy = tf.reduce_mean(
        tf.cast(corrct_prediction, tf.float32))  # 轉換張量float32

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})  # 求均值

    return result
# ------------------------------------------------------------------


# 導入圖片數據-----------------------------------------------------------
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
# ------------------------------------------------------------------


# 搭建網路-----------------------------------------------------------
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # 定義輸出層

# the error between prediction and real data
cross_entropy = tf.reduce_mean(tf.reduce_sum(ys *
                               tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()  # 載入方法

# 使用 sess 這個 session 執行
sess.run(tf.initialize_all_variables())  # 初始化
# ------------------------------------------------------------------


# 訓練--------------------------------------------------------------
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

if i % 50 == 0:
    print(compute_accuracy(
        mnist.test.images, mnist.test.labels
    ))

# 下列是輸出誤差
# if i % 50 == 0:
#      # to see the step improvement
#     print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
# ------------------------------------------------------------------

# 出來的結果會是辨識的正確率

# ------------------------------------------------------------------
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/557642/
# https://mofanpy.com/tutorials/machine-learning/tensorflow/create-NN/
# https://weikaiwei.com/tf/tensorflow-mnist/
# https://www.itread01.com/content/1546903265.html
