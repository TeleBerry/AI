# import tensorflow as tf
from tensorflow import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1, out_size]), 0.1, name='b')
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, W), b
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):

    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    corrct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(corrct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer, softmax通常用於做classification
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()

# important step
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
if i % 50 == 0:
    print(compute_accuracy(
        mnist.test.images, mnist.test.labels
    ))


# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/557642/
# https://weikaiwei.com/tf/tensorflow-mnist/
