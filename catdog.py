# ---------準備資料--------------------------------------------------
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras_tqdm import TQDMNotebookCallback
from keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import shutil
from random import shuffle
from tqdm import tqdm_notebook
import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
# ---------準備資料--------------------------------------------------


# ---------建立引數--------------------------------------------------

def organize_datasrts(path_to_data, n=4000, ratio=0.2):
    files = os.listdir(path_to_data)
    files-[os.path.join(path_to_data, f)for f in files]
    shuffle(files)
    files = files[:n]

    n = int(len(files)*ratio)
    val, train = files[:n], files[n:]

    shutil.rmtree('./data/')
    print('/data/ removed')

    for c in ['dog', 'cat']:
        os.makedirs('./data/train/{0}/'.format(c))
        os.makedirs('./data/validation/{0}/'.format(c))

    print('folders created !')

    for t in tqdm_notebook(train):
        if 'cat' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'cats'))
        else:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'dogs'))

    for v in tqdm_notebook(val):
        if 'cat' in v:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'cats'))
        else:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'dogs'))

    print('data copied !')


batch_size = 32

# 標準化處理
train_datagen = ImageDataGenerator(
    rescale=1/255.,  # rescale放縮因子，調整像素
    shear_range=0.2,  # 錯切變換，讓座標保持不變
    horizontal_flip=True  # 水平翻轉
)

val_datagen = ImageDataGenerator(rescale=1/255.)

# ---------建立引數--------------------------------------------------

# 生成器，要讓資料增強
train_generator = train_datagen.flow_from_directory(
    './data/train/', target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    './data/train/', target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

# Found 20000 images belonging to 2 classes.
# Found 5000 images belonging to 2 classes.


# 卷積/池化層(3)---------------------------------------------------------------------------------------------------

model = Sequential()

model.add(Conv2D(32, (3, 3), inpur_shape=(150, 150, 3),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 全連接層(2)--------------------------------------------------------------
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 隨機梯度下降法-----------------------------------------------------------------------
epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
mosel.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Callback for loss logging per ephoch


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()

# Callback for early stopping the training
early_stopping-keras.callbacks.EarlyStopping
(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
