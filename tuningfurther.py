
# coding: utf-8

# In[3]:

import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import sys
import gc
import json
import configparser
from datetime import datetime
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import fbeta_score, precision_score, recall_score

from utils.f2thresholdfinder import *
from utils.loaderjpg import *
from utils.generator import *
from utils.custommetrics import *
from utils.visualization import *
from utils.predictorjpg import *
from utils.file import *

from pretrained.vgg16 import *
from pretrained.vgg19 import *
from pretrained.resnet50 import *
from pretrained.densenet121 import *
from pretrained.custommodels import *


# In[4]:

# command line args processing "python tuningfurther.py cfg/3.cfg"
config_file = sys.argv[1]

print('reading configurations from config file: {}'.format(config_file))

settings = configparser.ConfigParser()
settings.read(config_file)
data_dir = settings.get('data', 'data_dir')

rescaled_dim = 224

model_name = settings.get('model', 'name')
print('model: {}'.format(model_name))

pretrained_model_file = None
model_weights_file = None

#pretrained_model_file = 'D:/Downloads/amazon/bottleneck/resnet50/frozen164_20170703-204725.h5'
if settings.has_option('model', 'source'):
    pretrained_model_file = settings.get('model', 'source')
    print('source model file: {}'.format(pretrained_model_file))
elif settings.has_option('model', 'weights'):
    model_weights_file = settings.get('model', 'weights')
    print('model weights file: {}'.format(model_weights_file))
#else:
#    raise ValueError('Error: config file must have model file path or model weights file path')
    
frozen_layers = settings.getint('model', 'frozen_layers')
print('number of frozen layers: {}'.format(frozen_layers))

learning_rate_schedule = json.loads(settings.get('model', 'learning_rate_schedule'))
print(learning_rate_schedule)
max_epoch_per_learning_rate = json.loads(settings.get('model', 'max_epoch_per_learning_rate'))
print(max_epoch_per_learning_rate)

batch_size = settings.getint('model', 'batch_size') 
print('batch size: {}'.format(batch_size))
#batch_size = 64

file_uuid = time.strftime("%Y%m%d-%H%M%S")

verbose_level = 0

labels = ['slash_burn', 'clear', 'blooming', 'primary', 'cloudy', 'conventional_mine', 'water', 'haze', 'cultivation', 'partly_cloudy', 'artisinal_mine', 'habitation', 'bare_ground', 'blow_down', 'agriculture', 'road', 'selective_logging']


# In[5]:

df_train = pd.read_csv(data_dir + 'train_v2.csv')
x_train, y_train = load_training_set(df_train, rescaled_dim)
print(x_train.shape)
print(y_train.shape)


# In[6]:

number_of_samples = x_train.shape[0]
split = int(number_of_samples * 0.90)
                     
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

number_validations = number_of_samples - split


# In[7]:

# early stopping prevents overfitting on training data
early_stop = EarlyStopping(monitor='val_loss',patience=1, min_delta=0, verbose=0, mode='auto')


# In[8]:

img_normalization = image_normalization_func(model_name)

train_datagen = BottleNeckImgGenerator(normalization=img_normalization)
train_gen = train_datagen.trainGen(x_train, y_train, batch_size)
valid_datagen = BottleNeckImgGenerator(normalization=img_normalization)
valid_gen = valid_datagen.validationGen(x_valid, y_valid, batch_size)


# In[9]:

# previously top classfier trained to val_loss = 0.105
# model = load_model('D:/Downloads/amazon/bottleneck/resnet50/frozen175_20170703-110325.h5')

if model_name == 'vgg19':
    model = custom_top_model(model_name, num_classes=17, num_frozen_layers=frozen_layers)
else:
    # Load full model file or weight-only file. Addresses bug with Keras load_model for models with custom layers.
    if pretrained_model_file is not None:
        model = load_model(pretrained_model_file)
        model = freeze_layers(model, num_frozen_layers=frozen_layers)
    else:
        model = custom_top_model(model_name, num_classes=17, num_frozen_layers=frozen_layers)
        model.load_weights(model_weights_file)

print(model.summary())
# check trainability of all layers
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable if hasattr(layer, 'trainable') else False)


# In[ ]:

model_filepath = data_dir + 'bottleneck/{}/frozen{}_{}.h5'.format(model_name, frozen_layers, file_uuid)
# save only the best model, not the latest epoch model.
model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

# also save the weights because Keras raises error in load_model when model contains custom layers
weights_filepath = data_dir + 'bottleneck/{}/frozen{}_{}_weights_only.h5'.format(model_name, frozen_layers, file_uuid)
weights_checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)


# In[ ]:

training_start_time = datetime.now()

history = {}
f2_history = []

num_samples_per_epoch = x_train.shape[0]

for learn_rate, epochs in zip(learning_rate_schedule, max_epoch_per_learning_rate):
    print('learning rate :{}'.format(learn_rate))
    model.optimizer.lr.set_value(learn_rate)
    
    tmp_history = model.fit_generator(train_gen,
                        samples_per_epoch=num_samples_per_epoch,
                        nb_epoch=epochs,
                        validation_data=valid_gen,
                        nb_val_samples=number_validations,              
                        verbose=verbose_level,
                        callbacks=[early_stop, model_checkpoint, weights_checkpoint])
    
    for k, v in tmp_history.history.iteritems():
        history.setdefault(k, []).extend(v)

time_spent_trianing = datetime.now() - training_start_time
print('{} model training complete. Time taken: {}'.format(model_name, time_spent_trianing))


# In[ ]:

# warning: load_model does not work for nested models such as our custom VGG19
# load your best model before making any final predictions

import gc
del model
gc.collect()

if pretrained_model_file is not None:
    print('loading model file: {}'.format(model_filepath))
    model = load_model(model_filepath)
else:
    print('loading model weights file: {}'.format(weights_filepath))
    model = custom_top_model(model_name, num_classes=17, num_frozen_layers=frozen_layers)
    model.load_weights(weights_filepath)


# In[ ]:

valid_datagen = BottleNeckImgGenerator(normalization=img_normalization)

y_predictions, optimized_thresholds = predict_with_optimal_thresholds(x_valid, y_valid, valid_datagen, model)

# save for use in ensembling weight optimization!
np.save(data_dir + '/temp/{}_{}_valid_set_predictions'.format(model_name, file_uuid), y_predictions)


# In[ ]:

threshold_df = pd.DataFrame({'label':labels, 
                             'optimized_threshold':optimized_thresholds})
print(threshold_df)


# In[ ]:

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


# In[ ]:

figures_dir = 'figures/{}'.format(model_name)
makedirs(figures_dir)

plot_file_path = figures_dir + '/stats_frozen{}_{}.png'.format(frozen_layers, file_uuid)
trainHistoryPlot(plot_file_path, history, f2_history, prediction_stats_df)


# In[ ]:

sample_submission_filepath = data_dir + 'sample_submission_v2.csv'

real_submission_filepath = data_dir + 'my_submissions/submission_{}_{}.csv'.format(model_name, file_uuid)

img_normalization = image_normalization_func(model_name)
test_datagen = BottleNeckImgGenerator(normalization=img_normalization)


# In[ ]:

make_submission(model,
                optimized_thresholds,
                rescaled_dim, 
                labels,
                sample_submission_filepath,
                real_submission_filepath,
                test_datagen)


# In[ ]:



