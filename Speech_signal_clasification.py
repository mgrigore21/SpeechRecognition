
# Part of code for my MSc Thesis Speech Recognition Methods
# Summary: Use of speech recognition database from Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017
#          Extract spectrograms for each recording
#          Train CNN with spectograms so that accuracy is >90 for 20 classes of commands
#          After 70 epochs, 95% accuracy, Succes!

# coding: utf-8

# # Speech Recognition using CNN  #

# In[ ]:


import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

import glob

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.io import wavfile as wav

import gc


# In[ ]:


# function to create spectrogram
# https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# In[ ]:


# read from files and create spectrogram
# split spectograms in train and validation folders
# creates folder structure for Keras part

root='../original_wav_input/train/audio/'
#all the categories in the audio folder
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
dirlist=['two','up','yes','zero']
for item in dirlist:
    train_audio_path=root+item
    audio_list=os.listdir(train_audio_path)
    number_audio_files=len(audio_list)
    if number_audio_files > 2299:
        os.mkdir('../data/train/'+item)
        os.mkdir('../data/validation/'+item)
        print(train_audio_path)
        
        #all the wav files in each folder 
        files=glob.glob(train_audio_path+'/*.wav')
        i=0
        for name in files:
            i=i+1
            sample_rate, samples = wavfile.read(name)
            freqs, times, spectrogram = log_specgram(samples, sample_rate)
            plt.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
            plt.axis('off')
            if i<2001:
                plt.savefig('../data/train/'+item+'/'+str(i),dpi=30,bbox_inches='tight')
            if i>2000:
                if i<2301:
                    plt.savefig('../data/validation/'+item+'/'+str(i),dpi=30,bbox_inches='tight')
            plt.close('all')
            gc.collect(2)


# In[1]:

# CNN part

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images.
img_width=162
img_height=107

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 2000
nb_validation_samples = 300
epochs=70
batch_size=16


# In[2]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height,3)


# In[3]:


model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(128,(3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

print(model.summary())


# In[4]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[5]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


