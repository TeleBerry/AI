import numpy as np
import tensorflow as tf

mple = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })  # 將 image數據和 label取出来

img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [128, 128, 3])  # reshape為 128*128的 3通道圖片
img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 丟出 img張量
label = tf.cast(features['label'], tf.int32)  # 丟出 label張量

return img, label
