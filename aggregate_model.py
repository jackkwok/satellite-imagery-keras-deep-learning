
# coding: utf-8

# In[1]:

import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm

from utils.file import makedirs
from utils.recorder import record_model_medata, record_model_scores
from utils.loader import load_training_set, load_test_set
from utils.f2thresholdfinder import *
from utils.imagegen import *
from utils.models import *
from utils.custommetrics import *

from datetime import datetime
import time
import configparser
import json
import sys


# In[2]:

timestr = time.strftime("%Y%m%d-%H%M%S")
start_time = datetime.now()


# In[3]:

config_file = 'cfg/default.cfg'

# e.g. > python aggregate_model.py cfg/3.cfg
if len(sys.argv) > 1 and '.cfg' in sys.argv[1]:
    config_file = sys.argv[1]

print('reading configurations from config file: {}'.format(config_file))

settings = configparser.ConfigParser()
settings.read(config_file)
data_dir = settings.get('data', 'data_dir')

df_train = pd.read_csv(data_dir + 'train_v2.csv')
model_filename = 'aggregate_model_'+ timestr +'.h5'
model_filepath = data_dir + 'models/' + model_filename
sample_submission_filepath = data_dir + 'sample_submission_v2.csv'
number_of_samples = len(df_train.index)
print('total number of training samples: {}'.format(number_of_samples))

# WARNING: keras allow either 1, 3, or 4 channels per pixel. Other numbers not allowed.
data_mask_label = np.array(['R', 'G', 'B', 'NDVI', 'NDWI', 'NIR'])
#print(settings.get('data', 'data_mask'))
data_mask_list = json.loads(settings.get('data', 'data_mask'))

data_mask = ma.make_mask(data_mask_list)
print(data_mask)

num_channels = np.sum(data_mask)
need_norm_stats = False

model_id = settings.get('model', 'model_id')
print('model: {}'.format(model_id))

# TODO understand the implications of this. this number usually does not include augmented images but:
# +0.01 F2 score improvement when number_of_samples * 3
num_samples_per_epoch = number_of_samples # number_of_samples * 3 

# default to 64
rescaled_dim = 64
if settings.has_option('data', 'rescaled_dim'):
    rescaled_dim = settings.getint('data', 'rescaled_dim')
print('rescaled dimension: {}'.format(rescaled_dim))

# Per Keras FAQ, one epoch is defined as an arbitrary cutoff that is one pass over the entire training set
number_epoch = settings.getint('model', 'number_epoch')

# a batch results in exactly one update to the model.
# batch_size is limited by model size and GPU memory
batch_size = settings.getint('model', 'batch_size') 
print('batch size: {}'.format(batch_size))

classifier_threshold = 0.2

split = int(number_of_samples * 0.80)  # TODO we may want to increase to 0.90 eventually
number_validations = number_of_samples - split


# In[6]:

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

print(labels)
print(len(labels))


# In[7]:

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}


# In[8]:

x_train, y_train = load_training_set(df_train, rescaled_dim)
print(x_train.shape)
print(y_train.shape)


# In[9]:

x_train = x_train[:, :, :, data_mask]

x_train = x_train.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
print(x_train.shape)


# In[10]:

# shuffle the samples because 
# 1) the original samples may not be randomized & 
# 2) to avoid the possiblility of overfitting the validation data while we tune the model
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=0)

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]


# In[11]:

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# In[12]:

image_generator = ScaledDown() # NormalizedByFeature() offers seemingly no improvement but cost 30 min more to run.


# In[13]:

# this is the augmentation configuration we will use for training
# TODO augment with random rotations for rare classes
train_datagen = image_generator.getTrainGenenerator()


# In[14]:

if (need_norm_stats):
    # need to compute internal stats like featurewise std and zca whitening
    train_datagen.fit(x_train)


# In[15]:

train_generator = train_datagen.flow(
        x_train, 
        y_train, 
        batch_size=batch_size,
        shuffle=True) 


# In[16]:

validation_datagen = image_generator.getValidationGenenerator()


# In[17]:

# workaround to provide your own stats: 
# http://stackoverflow.com/questions/41855512/how-does-data-normalization-work-in-keras-during-prediction/43069409#43069409
if (need_norm_stats):
    # need to compute internal stats like featurewise std and zca whitening
    validation_datagen.fit(x_valid)


# In[18]:

validation_generator = validation_datagen.flow(
        x_valid,
        y_valid,
        batch_size=batch_size,
        shuffle=False)


# In[19]:

model = get_model(model_id, num_channels, rescaled_dim, rescaled_dim)

# TODO 
# Use custom loss function to optimize F2 score.
# https://github.com/fchollet/keras/issues/369
# https://github.com/fchollet/keras/blob/master/keras/losses.py
model.compile(loss='binary_crossentropy', # Is this the best loss function?
              optimizer='adam',
              metrics=['accuracy', 'recall', 'precision'])


# In[20]:

# BUG when resuming training, the learning rate need to be decreased.
# let's load an existing trained model and continue training more epoch gives 0.01 improvement in LB score.
# model = load_model(data_dir + 'models/aggregate_model_20170507-124128.h5') # 0.86
# model = load_model(data_dir + 'models/aggregate_model_20170507-184232.h5') # 0.87
# model = load_model(data_dir + 'models/aggregate_model_20170511-133235.h5')
# model = load_model(data_dir + 'models/aggregate_model_20170515-062741.h5')
#number_epoch = 2


# In[21]:

# Ran into MemoryError when training DAGG_2 with 4 channels at epoch 50.
# To try to get reduce memory usage, we are limited the number of samples and batch_size

validation_num_samples = min(1280, number_of_samples - split)
x_valid_f2 = x_valid[:validation_num_samples]
y_valid_f2 = y_valid[:validation_num_samples]

# Note: threshold is fixed (not optimized per label)
def compute_f2_measure(l_model):    
    val_generator_f2 = validation_datagen.flow(
        x_valid_f2,
        y_valid_f2,
        batch_size=128,
        shuffle=False)
    raw_pred = l_model.predict_generator(val_generator_f2, validation_num_samples)
    thresholded_pred = (np.array(raw_pred) > classifier_threshold).astype(int)
    l_f2_score = fbeta_score(y_valid_f2, thresholded_pred, beta=2, average='samples')
    return l_f2_score
    
class F2_Validation(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f2_measures = []
    def on_epoch_end(self, epoch, logs={}):
        self.f2_measures.append(compute_f2_measure(self.model))

f2_score_val = F2_Validation()


# In[22]:

# stop overfitting on training data
early_stop = EarlyStopping(monitor='val_loss',patience=3, min_delta=0, verbose=0, mode='auto')  # TODO patience and min_delta


# In[23]:

training_start_time = datetime.now()
# fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_generator,
                    samples_per_epoch=num_samples_per_epoch,
                    nb_epoch=number_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=number_validations,
                    callbacks=[f2_score_val, early_stop])

model.save(model_filepath)  # always save your model and weights after training or during training
time_spent_trianing = datetime.now() - training_start_time

print('model training complete')


# In[ ]:

print(model.summary())


# In[24]:


# model = load_model(data_dir + 'models/aggregate_model_20170517-062305.h5')


# In[40]:

# use the validation data to compute some stats which tell us how the model is performing on the validation data set.
val_generator_score_board = validation_datagen.flow(
    x_valid,
    y_valid,
    batch_size=batch_size,
    shuffle=False)
p_valid = model.predict_generator(val_generator_score_board, number_validations)

#print(y_valid)
#print(p_valid)

optimized_thresholds = f2_optimized_thresholds(y_valid, np.array(p_valid))

y_predictions = (np.array(p_valid) > optimized_thresholds).astype(int)
#print(y_predictions)

# F2 score, which gives twice the weight to recall emphasising recall higher than precision
# 'samples' is what the evaluation criteria is for the contest
f2_score = fbeta_score(y_valid, y_predictions, beta=2, average='samples')
print('>>>> Overall F2 score over validation set using samples averaging ' , f2_score)


# In[26]:

threshold_df = pd.DataFrame({'label':labels, 
                             'optimized_threshold':optimized_thresholds})
print(threshold_df)


# In[33]:

precision_l, recall_l, f2_score_l = calculate_stats_for_prediction(y_valid, y_predictions)

count_stats_df = pd.DataFrame({
    'label': labels, 
    'true_sum': np.sum(y_valid, axis=0),
    'predict_sum': np.sum(y_predictions, axis=0),
    'f2': f2_score_l,
    'recall': recall_l,
    'precision': precision_l
})

# reordering the columns for easier reading
count_stats_df = count_stats_df[['label', 'f2', 'recall', 'precision', 'true_sum', 'predict_sum']]
print(count_stats_df)


# In[ ]:

filtered_data_mask_label = data_mask_label[data_mask]

record_model_scores(model_filename, 
                    model_id, 
                    history, 
                    f2_score, 
                    time_spent_trianing, 
                    num_channels,
                    config_file,
                    np.array_str(filtered_data_mask_label))


# In[23]:

figures_dir = 'figures/' + model_id
makedirs(figures_dir)

# list all data in history
print('training history stats:')
print(history.history.keys())

# summarize history for f2 score
fig = plt.figure(figsize=(15, 10))
subplot0 = fig.add_subplot(231)
if hasattr(f2_score_val, 'f2_measures'):
    subplot0.plot(f2_score_val.f2_measures)
subplot0.set_title('f2 score')
subplot0.set_ylabel('f2 score')
subplot0.set_xlabel('epoch')
subplot0.legend(['val'], loc='upper left')

# summarize history for recall
subplot3 = fig.add_subplot(232)
subplot3.plot(history.history['recall'])
subplot3.plot(history.history['val_recall'])
subplot3.set_title('recall')
subplot3.set_ylabel('recall')
subplot3.set_xlabel('epoch')
subplot3.legend(['train', 'val'], loc='upper left')

# summarize history for precision
subplot2 = fig.add_subplot(233)
subplot2.plot(history.history['precision'])
subplot2.plot(history.history['val_precision'])
subplot2.set_title('precision')
subplot2.set_ylabel('precision')
subplot2.set_xlabel('epoch')
subplot2.legend(['train', 'val'], loc='upper left')

# summarize history for accuracy
subplot1 = fig.add_subplot(234)
subplot1.plot(history.history['acc'])
subplot1.plot(history.history['val_acc'])
subplot1.set_title('accuracy')
subplot1.set_ylabel('accuracy')
subplot1.set_xlabel('epoch')
subplot1.legend(['train', 'val'], loc='upper left')

# summarize history for loss
subplot4 = fig.add_subplot(235)
subplot4.plot(history.history['loss'])
subplot4.plot(history.history['val_loss'])
subplot4.set_title('model loss')
subplot4.set_ylabel('loss')
subplot4.set_xlabel('epoch')
subplot4.legend(['train', 'val'], loc='upper left')

fig.savefig(figures_dir + '/stats_' + timestr + '.png')
#plt.show()


# In[24]:

#model = load_model(model_filepath)


# In[27]:

testset_dir = data_dir + 'test'

df_test_list = pd.read_csv(sample_submission_filepath)

x_test = load_test_set(df_test_list, rescaled_dim)

x_test = x_test[:, :, :, data_mask]


# In[28]:

#x_test = np.array(x_test, np.uint8)
print(x_test.shape)
x_test = x_test.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
print(x_test.shape)


# In[29]:

# this is the configuration we will use for testing:
testset_datagen = image_generator.getTestGenenerator()

if (need_norm_stats):
    # need to compute internal stats like featurewise std and zca whitening
    testset_datagen.fit(x_test)


# In[30]:

testset_generator = testset_datagen.flow(
    x_test,
    y=None,
    batch_size=batch_size,
    shuffle=False)
    
# ??? There may be a bug below that casues LB score to be 0.5-0.6
# testset_generator = testset_datagen.flow_from_directory(
#         testset_dir,
#         target_size=(rescaled_dim, rescaled_dim),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False)


# In[31]:

#from keras.models import load_model
# model = load_model(data_dir + 'models/aggregate_model_20170507-184232.h5')
# model = load_model(data_dir + 'models/aggregate_model_20170509-215809.h5')
# model = load_model(data_dir + 'models/aggregate_model_20170511-001322.h5')
# model = load_model(data_dir + 'models/aggregate_model_20170511-150149.h5')


# In[32]:

# run predictions on test set
testset_predict = model.predict_generator(testset_generator, x_test.shape[0]) # number of test samples

y_testset_predictions = (np.array(testset_predict) > optimized_thresholds).astype(int)

result = pd.DataFrame(y_testset_predictions, columns = labels)

preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.transpose()
    a = a.loc[a[i] == 1]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test = pd.read_csv(sample_submission_filepath)
df_test['tags'] = preds
df_test
print('done')


# In[33]:

#test code
# nums_ones = np.ones((1, 17))
# nums_zeros = np.zeros((1, 17))
# haha = np.array([[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]])

# y_testset_predictions = haha
# result = pd.DataFrame(y_testset_predictions, columns = labels)

# preds = []
# for i in tqdm(range(result.shape[0]), miniters=1000):
#     a = result.ix[[i]]
#     #print(a)
#     a = a.transpose()
#     print(a)
#     a = a.loc[a[i] == 1]
#     print(a)
#     ' '.join(list(a.index))
#     preds.append(' '.join(list(a.index)))
    
# print(preds)


# In[34]:

df_test.to_csv(data_dir + 'my_submissions/submission_' + timestr + '.csv', index=False)


# In[35]:

total_exec_time = datetime.now() - start_time
print ('time spent to complete execution: {}'.format(total_exec_time))


# In[ ]:



