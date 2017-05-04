
# coding: utf-8

# In[58]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import cv2
from tqdm import tqdm


# In[ ]:




# In[59]:

x_train = []
x_test = []
y_train = []

data_dir = 'D:/Downloads/amazon/'

df_train = pd.read_csv(data_dir + 'train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

print(labels)
print(len(labels))


# In[60]:

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}


# In[61]:

number_of_samples = 40000
split = 35000
num_samples_per_epoch = 120000 #  increase training samples at every epoch.

for f, tags in tqdm(df_train.values[:number_of_samples], miniters=1000):
    img = cv2.imread(data_dir + 'train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)


# In[62]:

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.uint8)

print(x_train.shape)
print(y_train.shape)


# In[63]:

x_train = x_train.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
print(x_train.shape)


# In[64]:

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]


# In[65]:

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# In[66]:

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True)


# In[67]:

train_generator = train_datagen.flow(
        x_train, 
        y_train, 
        batch_size=32,
        shuffle=True) 


# In[68]:

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False,
        vertical_flip=False)

validation_generator = validation_datagen.flow(
        x_valid, 
        y_valid, 
        batch_size=32,
        shuffle=False)


# In[ ]:

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32)))  
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_generator,
                    samples_per_epoch=num_samples_per_epoch,
                    nb_epoch=8,
                    verbose=1)
model.save(data_dir + 'models/aggregate_model_001.h5')  # always save your model and weights after training or during training
print('done')


# In[ ]:

from sklearn.metrics import fbeta_score
#np.set_printoptions(threshold='nan')

# validation images has not been normalized!
#p_valid = model.predict(x_valid / float(255), batch_size=128)
p_valid = model.predict_generator(validation_generator, number_of_samples - split)

print(y_valid)
print(p_valid)

y_predictions = (np.array(p_valid) > 0.2).astype(int)
print(y_predictions)

print(np.sum(y_valid, axis=0))
print(np.sum(y_predictions, axis=0))

print(fbeta_score(y_valid, y_predictions, beta=2, average='samples'))

#TODO calculate f2 score for each label. find out which labels model is predicting badly.


# In[ ]:

def f2score(truth, predict, label_index):
    score = fbeta_score(truth[:, label_index], predict[:, label_index], beta=2, average='samples')
    return score
    
for x in range(0, len(labels)):
    score = f2score(y_valid, y_predictions, x)
    label = labels[x]
    print(label, ' (f2 score): ' , score)


# In[ ]:



