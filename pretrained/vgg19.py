# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import load_model
from keras.models import Model

from sklearn.metrics import log_loss

imagenet_vgg_weights_file = 'D:/Downloads/amazon/imagenet_models/vgg19_weights.h5'

def vgg19_model(channel=3, num_classes=17):
    """
    VGG 19 Model for Keras

    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing

    Parameters:
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
  
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(channel, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights(imagenet_vgg_weights_file)
    return model

def vgg19_model_fc_truncated (channel=3):
    """Used to generate bottleneck features with ImageNet weigths loaded. """
    model = vgg19_model(channel=channel)
    # Truncate the fully connected layers up to flatten layer
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    return model

# Dense(4096), 0.5, Dense(4096), 0.5 val_loss=0.1228
# Dense(4096), 0.25, Dense(2048), 0.25 val_loss=0.1240
# Dense(2048), 0.25, Dense(1024), 0.25 val_loss=0.1223
# Dense(1024), 0.25, Dense(512), 0.25 val_loss=0.1228
# Dense(512), 0.25, Dense(256), 0.25 val_loss=0.1228
# Dense(1024), 0.25, Dense(256), 0.25 val_loss=0.1238
# Dense(256), 0.25, Dense(128), 0.25 val_loss=0.1254
# Dense(2048), 0.25 val_loss=0.1269
# Dense(4096), 0.25 val_loss=0.1259
def vgg19_custom_top_classifier(input_shape, num_classes=None):
    """
    Parameters:
    input_shape: the shape of the bottleneck features """
    model = Sequential()
    model.add(Dense(2048, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='sigmoid'))  # softmax replace with sigmoid for multiclass multlabel classification
    return model

def vgg19_model_custom_top(channel=3, num_classes=17, num_frozen_layers=38):
    """
    WARNING: nested model will prevent Keras from loading model file and weight file properly.
    I am going with this route because for unknown reason the val_loss is super high going the regular route.

    Parameters:
    input_shape: the shape of the bottleneck features : bottleneck_training_data.shape[1:] 
    """
    model = vgg19_model_fc_truncated(channel)

    # add custom layers as determined by experiments to guess best architecture
    # model.add(Dense(2048, activation='relu', input_shape=model.output.shape))
    # model.add(Dropout(0.25))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(num_classes, activation='sigmoid'))
    custom_fc_model_path = 'D:/Downloads/amazon/bottleneck/vgg19/bottleneck_fc_model_val_loss_0122.h5'
    top_model = load_model(custom_fc_model_path)
    model.add(top_model)

    freeze_layers(model, num_frozen_layers=num_frozen_layers)
    return model

def freeze_layers(model, num_frozen_layers=38):
    """num_frozen_layers: number of frozen layers total counting from bottom.  0 indexed."""
    for layer in model.layers[num_frozen_layers:]:
        if hasattr(layer, 'trainable'):
            layer.trainable = True

    for layer in model.layers[:num_frozen_layers]:
        if hasattr(layer, 'trainable'):
            layer.trainable = False
    
    optimizer = Adam(lr=0.0001)
    #optimizer = SGD(lr=0.001, nesterov=True)
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])
    return model
