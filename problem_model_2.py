import os
import numpy as np
import keras.utils
from PIL import Image

size = (128,128) #強制讓圖片符合大小
normal_img_list=[] #空變數好放資料
error_img_list=[] #..

base_path = r'D:\Teleberry_01\Problem_img' 

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".jpg"):
            filename = os.path.join(root, file)
            file_size = os.path.getsize(filename)
            category_name = os.path.basename(root) #把最後一層資料夾的名稱取出來，用來把貓跟狗的照片放進不同的變數當中
            if file_size >=1000: #圖片大於1kb
                im = Image.open(filename)
                if im.mode=='RGB':
                    im=im.resize(size,Image.BILINEAR)
                    imarray = np.array(im)
                    imarray = (imarray - np.min(imarray))/(np.max(imarray)-np.min(imarray))
                    if category_name == 'normal':
                        normal_img_list.append(imarray)
                    elif category_name == "error":
                        error_img_list.append(imarray) 
                        
normal_img_arr = np.asarray(normal_img_list) #把list整理成array
error_img_arr = np.asarray(error_img_list) 

normal_img_label = np.ones(normal_img_arr.shape[0])*0  #製作標籤normal 0
error_img_label = np.ones(error_img_arr.shape[0])*1  #製作標籤error 1

img_arr   = np.concatenate((normal_img_arr, error_img_arr), axis = 0) #normal和error合在一起的總變數
img_label = np.concatenate((normal_img_label, error_img_label), axis = 0) 
img_label = keras.utils.to_categorical(img_label, num_classes = 2)
# print(img_label)                        

import random
temp = list(zip(img_arr, img_label))
random.shuffle(temp)
img_arr, img_label = zip(*temp)
img_arr=np.asarray(img_arr)
img_label=np.asarray(img_label)
del temp
# print(img_label)

from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(img_arr, img_label, test_size=0.2, random_state=42)

#%% Create Model
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout2D, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.callbacks import EarlyStopping

# Generate model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,3),padding='same',name='block1_conv2_1'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
model.add(SpatialDropout2D(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
model.add(SpatialDropout2D(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu',name='final_output_1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu',name='final_output_2'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid',name='class_output'))
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
EStop = EarlyStopping(monitor='val_acc', min_delta=0, 
                      patience=10, verbose=1, mode='auto')

#%% Training and saving
history = model.fit(train_data, train_label, batch_size=64, epochs=50,shuffle=True, validation_split=0.2,callbacks=[EStop])

import time
timestr = time.strftime("%Y%m%d_%H%M%S")
model.save('problem_model_{}.h5'.format(timestr)) 

# Model Structure
from keras.utils import plot_model
plot_model(model, to_file='model_{}.png'.format(timestr),show_shapes=True, show_layer_names=True)


