# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:49:04 2021

@author: YIHSUAN
"""
#%%
from PIL import Image
img=Image.open('PetImages/Cat/500.jpg')
#%%
import os
import glob
import numpy as np
from keras.preprocessing.image import  img_to_array, load_img
from PIL import Image

dict_labels = {"Cat":0, "Dog":1}
size = (64,64)
nbofdata=500   

for folders in glob.glob("PetImages/*"):
    print(folders)
    images=[]
    labels_hot=[]
    labels=[]
    nbofdata_i=1
    for filename in os.listdir(folders):
        if nbofdata_i <= nbofdata:
                    label = os.path.basename(folders)
                    className = np.asarray(label)
                    img=load_img(os.path.join(folders,filename))
                    img=img.resize(size,Image.BILINEAR)
                    if img is not None:
                        if label is not None:
                            labels.append(className)
                            labels_hot.append(dict_labels[label])
                        x=img_to_array(img)
                        images.append(x)
                    nbofdata_i+=1
    images=np.array(images)    
    labels_hot=np.array(labels_hot)


    print("images.shape={}, labels_hot.shape=={}".format(images.shape, labels_hot.shape))    
    imagesavepath='Cat_Dog_Dataset/'
    
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    np.save(imagesavepath+'{}_images.npy'.format(label),images)    
    np.save(imagesavepath+'{}_label.npy'.format(label),labels)    
    np.save(imagesavepath+'{}_labels_hot.npy'.format(label),labels_hot)
    print('{} files has been saved.'.format(label))