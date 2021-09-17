# ---------keras資料下載--------------------------------------------------
import kears
from kears.preprocessing.image import ImageDataGenerator
from kears_tqdm import TQDMNotebookCallback
from kears.models import Sequential
from kears.layers import Dense
from kears.layers import Dropout
from kears.layers import Flatten
from kears.constraints import maxnorm
from kears.optimizers import SGD
from kears.layers.convolutional import Conv2D
from kears.layers.convolutional import MaxPooling2D
from kears.utils import np_utils
from kears.callbacks import Callback
# ---------keras資料下載--------------------------------------------------

# ---------準備資料--------------------------------------------------
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import shutil
import pandas as pd

%matplotlib inline

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

train_datagen = ImageDataGenerator(
    rescale=1/255., shear_range=0.2, horizontal_filp=True)

val_datagen = ImageDataGenerator(rescale=1/255.)

# ---------建立引數--------------------------------------------------

# ---------資料增強--------------------------------------------------

# ---------資料增強--------------------------------------------------
