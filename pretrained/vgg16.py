# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import load_model

from sklearn.metrics import log_loss

imagenet_vgg_weights_file = 'D:/Downloads/amazon/imagenet_models/vgg16_weights.h5'

# Reference: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
# BGR mean values [103.94, 116.78, 123.68] should be subtracted before feeding into the model
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68 

def vgg16_model(channel=3):
	"""VGG 16 Model for Keras with ImageNet weights loaded.

	Model Schema is based on 
	https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

	ImageNet Pretrained Weights 
	https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

	Parameters:
	  img_rows, img_cols - resolution of inputs
	  channel - 1 for grayscale, 3 for color 
	  num_classes - number of categories for our classification task
	"""
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(channel, 224, 224)))  # is order correct?
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

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

def vgg16_model_fc_truncated (channel=3):
	"""Used to generate bottleneck features with ImageNet weigths loaded. """
	model = vgg16_model(channel)
	# Truncate the fully connected layers up to flatten layer
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()

	return model

def custom_fc_layers(input_shape, num_classes=None):
	"""input_shape: the shape of the bottleneck features : bottleneck_training_data.shape[1:] """
	model = Sequential()
	# Modification from VGG16: 4096 units may be overkill so using 1024
	model.add(Dense(4096, activation='relu', input_shape=input_shape))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replace with sigmoid for multiclass multlabel classification
	return model

def vgg16_model_custom_fc_layers(custom_fc_model_path, channel=3, num_classes=None, num_frozen_layers=25, learning_rate=1e-3):
	"""
	custom_fc_model_path: model with pre-trained weights from previous training of fully connected model
	input_shape: the shape of the bottleneck features : bottleneck_training_data.shape[1:] """
	model = vgg16_model_fc_truncated(channel)

	#model.outputs = [model.layers[-1].output]
	#model.layers[-1].outbound_nodes = []

	top_model = load_model(custom_fc_model_path)
	model.add(top_model)

	# set the bottom layers (up to the last conv block)
	# to non-trainable (i.e. weights will not be updated)
	for layer in model.layers[:num_frozen_layers]:
		layer.trainable = False

	# Learning rate is default to 0.001 (10x slower than default 0.01)
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])

	return model
