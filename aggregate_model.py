
# coding: utf-8

# In[1]:

import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import gc

import keras as k
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score, precision_score, recall_score

import cv2
from tqdm import tqdm

from datetime import datetime
import time
import configparser
import json
import sys

from utils.file import makedirs
from utils.recorder import record_model_medata, record_model_scores
from utils.loader import *
from utils.f2thresholdfinder import *
from utils.imagegen import *
from utils.models import *
from utils.custommetrics import *
from utils.samplesduplicator import duplicate_train_samples
from utils.training import *
from utils.predictor import *
from utils.augmentation import *
from utils.generator import *
from utils.visualization import *


# In[2]:

timestr = time.strftime("%Y%m%d-%H%M%S")
start_time = datetime.now()


# In[3]:

config_file = 'cfg/default.cfg'
#config_file = 'cfg/24.cfg'

# command line args processing "python aggregate_model.py cfg/3.cfg"
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
print('total number of samples: {}'.format(number_of_samples))

# WARNING: keras allow either 1, 3, or 4 channels per pixel. Other numbers not allowed.
data_mask_label = np.array(['R', 'G', 'B', 'NDVI', 'NDWI', 'NIR'])
#print(settings.get('data', 'data_mask'))
data_mask_list = json.loads(settings.get('data', 'data_mask'))

data_mask = ma.make_mask(data_mask_list)
print(data_mask)

num_channels = np.sum(data_mask)

model_id = settings.get('model', 'model_id')
print('model: {}'.format(model_id))

learning_rate_schedule = json.loads(settings.get('model', 'learning_rate_schedule'))
max_epoch_per_learning_rate = json.loads(settings.get('model', 'max_epoch_per_learning_rate'))
print('learning rates: {}'.format(learning_rate_schedule))
print('with max epoch: {}'.format(max_epoch_per_learning_rate))

# default to 64
rescaled_dim = 64
if settings.has_option('data', 'rescaled_dim'):
    rescaled_dim = settings.getint('data', 'rescaled_dim')
print('rescaled dimension: {}'.format(rescaled_dim))

# one epoch is an arbitrary cutoff : one pass over the entire training set

# a batch results in exactly one update to the model.
# batch_size is limited by model size and GPU memory
batch_size = settings.getint('model', 'batch_size') 
print('batch size: {}'.format(batch_size))

classifier_threshold = 0.2 # used for end of epoch f2 approximation only

split = int(number_of_samples * 0.90)  # Greater than 0.90 will result in inaccurate metrics for validation set
number_validations = number_of_samples - split


# In[4]:

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

print(labels)
print(len(labels))


# In[5]:

x_train, y_train = load_training_set(df_train, rescaled_dim)
print(x_train.shape)
print(y_train.shape)


# In[6]:

if x_train.shape[3] == 6:
    x_train = x_train[:, :, :, data_mask]

x_train = x_train.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
print(x_train.shape)


# In[7]:

# TODO if shuffle, save the shuffling order to hdf5 so we can recreate the training and validation sets post execution.
# shuffle the samples:
# 1) the original samples may not be randomized & 
# 2) to avoid the possiblility of overfitting the validation data while we tune the model
# from sklearn.utils import shuffle
# x_train, y_train = shuffle(x_train, y_train, random_state=0)

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]


# In[8]:

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# In[9]:

# experimental hack to get more samples for augmentations for a specific low-frequency tag in unbalanced dataset. e.g. habitation
# selecting the optimal multiplier is sensitive

# augmentation_hack_config = settings.has_section('augmentation_hack')
# if augmentation_hack_config:
#     dup_multiplier = settings.getint('augmentation_hack', 'multiplier')
#     hack_label_target = settings.get('augmentation_hack', 'label_target')

# if augmentation_hack_config:
#     x_train, y_train = duplicate_train_samples(x_train, y_train, labels.index(hack_label_target), multiplier=dup_multiplier)
#     print(x_train.shape)
#     print(y_train.shape)


# In[10]:

single_taget_model = False
# warning: experimental. 
# shuffling won't let you put things back together
if settings.has_option('data', 'single_target'):
    single_taget_model = True
    single_target_label = settings.get('data', 'single_target')
    single_target_label_index = labels.index(single_target_label)
    y_train = y_train[:,single_target_label_index]
    y_valid = y_valid[:,single_target_label_index]
    
score_averaging_method = 'binary' if single_taget_model else 'samples'
print('score_averaging_method', score_averaging_method)


# In[12]:

if single_taget_model:
    set_model_output_layer_size(1)
    
model = get_model(model_id, num_channels, rescaled_dim, rescaled_dim)


# In[13]:

# let's load an existing trained model and continue training more epoch gives 0.01 improvement in LB score.
# model = load_model(data_dir + 'models/aggregate_model_20170507-124128.h5') # 0.86
# model = load_model(data_dir + 'models/aggregate_model_20170507-184232.h5') # 0.87
# model = load_model(data_dir + 'models/aggregate_model_20170511-133235.h5')
# model = load_model(data_dir + 'models/aggregate_model_20170515-062741.h5')


# In[14]:

# Note: threshold is fixed (not optimized per label)
def compute_f2_measure(l_model, x_data, y_data):
    custom_val_gen = CustomImgGenerator()
    val_generator_f2 = custom_val_gen.validationGen(x_data, y_data, 64)
    raw_pred = l_model.predict_generator(val_generator_f2, x_data.shape[0])
    thresholded_pred = (np.array(raw_pred) > classifier_threshold).astype(int)
    l_f2_score = fbeta_score(y_data, thresholded_pred, beta=2, average=score_averaging_method)
    return l_f2_score

class F2_Validation(k.callbacks.Callback):
    def __init__(self, x_data, y_data):
        super(F2_Validation, self).__init__()
        # Ran into MemoryError when training DAGG_2 with 4 channels at epoch 50.
        # To try to get reduce memory usage, limit the number of samples to an arbitrary small number
        validation_num_samples = min(1280, x_data.shape[0])
        self.x_data = x_data[:validation_num_samples]
        self.y_data = y_data[:validation_num_samples]
    
    def on_train_begin(self, logs={}):
        self.f2_measures = []
    def on_epoch_end(self, epoch, logs={}):
        self.f2_measures.append(compute_f2_measure(self.model, self.x_data, self.y_data))


# In[16]:

# early stopping prevents overfitting on training data
early_stop = EarlyStopping(monitor='val_loss',patience=2, min_delta=0, verbose=0, mode='auto')

# save only the best model, not the latest epoch model.
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True)


# In[17]:

training_start_time = datetime.now()

history = {}
f2_history = []

custom_gen = CustomImgGenerator()

adam = optimizers.Adam()

# https://github.com/fchollet/keras/issues/369
# https://github.com/fchollet/keras/blob/master/keras/losses.py
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy', 'recall', 'precision'])

kf = KFold(n_splits=5, shuffle=True, random_state=1)

for learn_rate, epochs in zip(learning_rate_schedule, max_epoch_per_learning_rate):
    print('learning rate :{}'.format(learn_rate))
    model.optimizer.lr.set_value(learn_rate) # https://github.com/fchollet/keras/issues/888
    
    # use our custom generators to perform augmentation
    # TODO split x_train into smaller batches for larger models
    train_gen = custom_gen.trainGen(x_train, y_train, batch_size)
    valid_gen = custom_gen.validationGen(x_valid, y_valid, batch_size)
    
    f2_score_val = F2_Validation(x_valid, y_valid)
    
    # TODO understand the implications of num_samples_per_epoch.
    # +0.002 to +0.01??? F2 score improvement when number_of_samples * 3
    num_samples_per_epoch = x_train.shape[0]
    
    tmp_history = model.fit_generator(train_gen,
                        samples_per_epoch=num_samples_per_epoch,
                        nb_epoch=epochs,
                        validation_data=valid_gen,
                        nb_val_samples=number_validations,              
                        verbose=0,
                        callbacks=[f2_score_val, early_stop, checkpoint])
    
    for k, v in tmp_history.history.iteritems():
        history.setdefault(k, []).extend(v)
    print("f2 validation scores", f2_score_val.f2_measures)
    f2_history.extend(f2_score_val.f2_measures)


time_spent_trianing = datetime.now() - training_start_time
print('model training complete')


# In[18]:

print(f2_history)


# In[19]:

print(model.summary())


# In[20]:

# model = load_model(data_dir + 'models/aggregate_model_20170517-062305.h5')
print(y_valid.shape)
print(y_valid.ndim)


# In[21]:

# use the validation data to compute some stats which tell us how the model is performing on the validation data set.
val_generator_score_board = custom_gen.validationGen(x_valid, y_valid, batch_size)
p_valid = model.predict_generator(val_generator_score_board, number_validations)


# In[22]:

optimized_thresholds = f2_optimized_thresholds(y_valid, p_valid)

y_predictions = (np.array(p_valid) > optimized_thresholds).astype(int)

# save for use in ensembling weight optimization!
np.save(data_dir + '/temp/valid_set_predictions', y_predictions)

precision_s = precision_score(y_valid, y_predictions, average=score_averaging_method)
print('>>>> Overall precision score over validation set ' , precision_s)

recall_s = recall_score(y_valid, y_predictions, average=score_averaging_method)
print('>>>> Overall recall score over validation set ' , recall_s)

# F2 score, which gives twice the weight to recall
# 'samples' is what the evaluation criteria is for the contest
f2_score = fbeta_score(y_valid, y_predictions, beta=2, average=score_averaging_method)
print('>>>> Overall F2 score over validation set ' , f2_score)


# In[23]:

threshold_df = pd.DataFrame({'label':labels, 
                             'optimized_threshold':optimized_thresholds})
print(threshold_df)


# In[24]:

precision_l, recall_l, f2_score_l = calculate_stats_for_prediction(y_valid, y_predictions)

prediction_stats_df = pd.DataFrame({
    'label': labels, 
    'true_sum': np.sum(y_valid, axis=0),
    'predict_sum': np.sum(y_predictions, axis=0),
    'f2': f2_score_l,
    'recall': recall_l,
    'precision': precision_l
})

# reordering the columns for easier reading
prediction_stats_df = prediction_stats_df[['label', 'f2', 'recall', 'precision', 'true_sum', 'predict_sum']]
print(prediction_stats_df)


# In[25]:

print(history['val_acc'])


# In[26]:

filtered_data_mask_label = data_mask_label[data_mask]

record_model_scores(model_filename, 
                    model_id, 
                    history, 
                    f2_score, 
                    time_spent_trianing, 
                    num_channels,
                    config_file,
                    np.array_str(filtered_data_mask_label),
                    os.path.basename(get_training_set_file_path(rescaled_dim)))


# In[27]:

figures_dir = 'figures/' + model_id
makedirs(figures_dir)

print('training history dictionary keys:', history.keys())

plot_file_path = figures_dir + '/stats_' + timestr + '.png'
trainHistoryPlot(plot_file_path, history, f2_history, prediction_stats_df)


# In[5]:

if not is_test_set_in_cache(rescaled_dim):
    # populate the test dataset cache
    df_test = pd.read_csv(sample_submission_filepath)
    load_test_set(df_test, rescaled_dim)


# In[9]:

## delete me!
# optimized_thresholds = np.empty(17)
# optimized_thresholds.fill(0.2)
# print(optimized_thresholds)

# print(labels)

# print(data_mask)
# timestr = 'good_test_data'


# In[10]:

#model = load_model(data_dir + 'models/aggregate_model_20170625-111821.h5')


# In[11]:

real_submission_filepath = data_dir + 'my_submissions/submission_' + timestr + '.csv'
print(real_submission_filepath)


# In[12]:

make_submission(model,
                optimized_thresholds,
                data_mask,
                rescaled_dim, 
                labels, 
                sample_submission_filepath,
                real_submission_filepath)


# In[29]:

total_exec_time = datetime.now() - start_time
print ('time spent to complete execution: {}'.format(total_exec_time))

