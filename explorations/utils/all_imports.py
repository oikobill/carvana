#!/Users/Vasilis/anaconda3/bin/python

# A FILE THAT CONTAINS ALL THE IMPORT STATEMENTS I WILL EVER USE
import numpy as np
#import matplotlib.pyplot as plt
import os
from scipy.misc import imresize
from scipy.misc import imsave
from tqdm import tqdm
import pandas as pd

# Keras stuff
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# scikit-learn
from sklearn.model_selection import train_test_split
