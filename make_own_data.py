import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = './test/'  # 測試資料

classes = {'dog', 'cat'}  # 設定2類

writer = tf.compat.v1.python_io.TFRecordWriter(
    "dog_and_cat_test.tfrecords")  # 要生成的類別

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name  # 圖片的位址

        img = Image.open(img_path)
        img = img.resize((128, 128))  # 更改圖片大小
        print(np.shape(img))
        img_raw = img.tobytes()  # 將圖轉為二進制
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example對象對label和image數據進行封裝
        writer.write(example.SerializeToString())  # 序列化為字符串

writer.close()
