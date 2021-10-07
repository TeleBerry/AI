import os
import glob
import numpy as np
import keras
from keras.preprocessing.image import  img_to_array, load_img
from PIL import Image
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
#%%
dict_labels = {"Cat":0, "Dog":1}
size = (128,128) #由於原始資料影像大小不一，因此制定一個統一值
nbofdata=100   #從各個資料夾中抓取特定數量的檔案
#%% Read Traget Folders' Path
labels=['Cat','Dog']
base_path = r'D:\Teleberry_01\PetImages'
layers_of_folders=0
folder_list=[]
if base_path :
    folder_layers=[]
    files = os.scandir(base_path)
    #  Get the 1st layer of folder
    first_folder = []
    first_folder_kind = []
    for entry in files:
        if entry.is_dir():
            first_folder.append(entry.path)
            first_folder_kind.append(entry.name)
    folder_layers.append(first_folder_kind)
    folder_list.append(first_folder)
    #  Get the 2nd layer of folder
    second_folder = []
    if first_folder:
        second_folder = []
        second_folder_kind = []
        layers_of_folders+=1
        for fldr in first_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    second_folder.append(entry.path)
                    second_folder_kind.append(entry.name)
        second_folder_kind= second_folder_kind[0:int(len(second_folder_kind)/len(first_folder_kind))]
        folder_layers.append(second_folder_kind)
        folder_list.append(second_folder)
    #  Get the 3rd layer of folder
    third_folder = []
    if second_folder:
        third_folder = []
        third_folder_kind = []
        layers_of_folders+=1
        for fldr in second_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    third_folder.append(entry.path)
                    third_folder_kind.append(entry.name)
        third_folder_kind= third_folder_kind[0:int(len(third_folder_kind)/(len(second_folder_kind)*len(first_folder_kind)))]
        folder_list.append(third_folder)
    #  Get the 4th layer of folder
    forth_folder = []
    if third_folder:
        forth_folder = []
        forth_folder_kind = []
        layers_of_folders+=1
        for fldr in third_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    forth_folder.append(entry.path)
                    forth_folder_kind.append(entry.name)
        forth_folder_kind= forth_folder_kind[0:int(len(forth_folder_kind)/(len(third_folder_kind)*len(second_folder_kind)*len(first_folder_kind)))]
        folder_list.append(forth_folder)
     #  Get the 5th layer of folder
    if forth_folder:
        fifth_folder = []
        fifth_folder_kind = []
        layers_of_folders+=1
        for fldr in third_folder:
            files = os.scandir(fldr)
            for entry in files:
                if entry.is_dir():
                    fifth_folder.append(entry.path)
                    fifth_folder_kind.append(entry.name)
        fifth_folder_kind= fifth_folder_kind[0:int(len(fifth_folder_kind)/(len(forth_folder_kind)*len(third_folder_kind)*len(second_folder_kind)*len(first_folder_kind)))]
#%% Read Image Files (*.tif)
datanumber=nbofdata
blob=[]
blob_nparray=[]
image_data=[]
conc = 0
labels_dict={}
for entry1 in folder_list[layers_of_folders - 1]:
    blob = []
    cellname = os.path.basename(os.path.dirname(entry1))  # extract cell name
    # print(cellname)
    concnames = os.path.basename(entry1)  # extract concentration
    # print(concnames)
    if concnames in labels:
        labels_dict[conc] = concnames
        fnamelist = glob.glob(os.path.join(entry1, '*.jpg'))
        for filename in fnamelist[0:datanumber]:
            im = Image.open(filename)
            if im is not None:
                if im.mode=='RGB':
                    im=im.resize(size,Image.BILINEAR)
                    imarray = np.array(im)
                    blob.append(imarray)
        ind = np.reshape(np.arange(1, len(blob) + 1), (-1, 1))
        blob_nparray = np.reshape(np.asarray(blob), (len(blob), blob[1].size))
        blob_nparray = np.hstack((blob_nparray, ind, conc * np.ones((len(blob), 1))))
        image_data.append(np.asarray(blob_nparray, dtype=np.float32))
        print(concnames+'  finished!')
        conc += 1
#%%
for j in range(len(labels)):
    trytry=image_data[j][:]
# Prepare data
    LengthT = trytry.shape[0]
    trytry_index = trytry[...,-2:-1]
    trytry_label = trytry[...,-1:] #['Nega' for x in range(lengthN*4)] #Nega_data[...,-1:]
    trytry = trytry[...,:-2]
    
    # Normalize image by subtracting mean image
    trytry -= np.reshape(np.mean(trytry, axis=1), (-1,1))
    # Reshape images
    trytry = np.reshape(trytry, (trytry.shape[0],128,128,3))
    
#    # Rotate images
#    for i in range(3):
#        trytry[LengthT*(i+1):LengthT*(i+2)] = np.rot90(trytry[:LengthT], i+1, (1,2))
    # Add channel dimension to fit in Conv2D
    trytry = trytry.reshape(-1,128,128,3)
    np.random.shuffle(trytry)
    trytry_train_upto = round(trytry.shape[0] * 8 / 10)
    trytry_test_upto = trytry.shape[0]
    if j is 0:
        train_data = trytry[:trytry_train_upto]
        test_data = trytry[trytry_train_upto:trytry_test_upto]
        train_label = trytry_label[:trytry_train_upto]
        test_label = trytry_label[trytry_train_upto:trytry_test_upto]
        
    else:
        train_data = np.concatenate((train_data, 
                                     trytry[:trytry_train_upto]), axis=0)
        
        test_data = np.concatenate((test_data, 
                                    trytry[trytry_train_upto:trytry_test_upto]), axis=0)
        
        train_label = np.concatenate((train_label, 
                                     trytry_label[:trytry_train_upto]), axis=0)
        
        
        test_label = np.concatenate((test_label, 
                                    trytry_label[trytry_train_upto:trytry_test_upto]), axis=0)
        
test_label = keras.utils.to_categorical(test_label, num_classes=len(labels))
train_label = keras.utils.to_categorical(train_label, num_classes=len(labels))

#%% Shuffle data
import random
temp = list(zip(train_data, train_label))

random.shuffle(temp)

train_data,train_label = zip(*temp)

train_data=np.asarray(train_data)
train_label=np.asarray(train_label)

#%% VGG 16 only for classification
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
# Generate model
model = Sequential()
# input: 190x190 images with 3 channels -> (190, 190, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,3),padding='same',name='block1_conv2_1'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
model.add(Dropout(0.25))
#
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_1'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_2'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block3_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block4_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block5_MaxPooling'))
#model.add(Dropout(0.25))
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
#%% Start Traning Model
history = model.fit(train_data, train_label, batch_size=64, epochs=100,shuffle=True, validation_split=0.2,callbacks=[EStop])
#%%
model.save('catdog_model.h5') 
predictions=model.predict(test_data)
#%%
from keras.preprocessing.image import array_to_img
i=1
if predictions[i, 1]>= 0.5: 
    print('I am {:.2%} sure this is a Dog'.format(predictions[i][1]))
else: 
    print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][1]))
    
plt.imshow(array_to_img(test_data[i]))
plt.show()