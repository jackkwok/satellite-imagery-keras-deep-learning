import keras as k
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model_output_layer_size = 17

def get_model_output_layer_size():
	return model_output_layer_size

def set_model_output_layer_size(size):
	global model_output_layer_size
	model_output_layer_size = size

def get_model(model_id, num_channels, rescaled_dim1, rescaled_dim2):
	model_dict = {
		'JAGG_1': JAGG_1,
		'JAGG_1_1': JAGG_1_1,
		'JAGG_2': JAGG_2,
		'JAGG_2_BN': JAGG_2_BN,
		'JAGG_2_D4X': JAGG_2_D4X,
		'JAGG_3': JAGG_3,
		'JAGG_4': JAGG_4,
		'JAGG_4_2': JAGG_4_2,
		'EKAMI': EKAMI,
		'VGG_16': VGG_16
	}
	return model_dict[model_id](num_channels, rescaled_dim1, rescaled_dim2)

def JAGG_1(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# removed 2 pooling layers from JAGG_1
def JAGG_1_1(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# double the capacity of JAGG_1
# Best Model
def JAGG_2(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# 4X the number of neurons in the Dense layer after flattening the final conv layer
# Added dropouts after each conv block
# No improvement over JAGG_2
def JAGG_2_D4X(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# Batch Norm is horribly slow (10x slower). Do not use Batch Norm.
def JAGG_2_BN(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))
	model.add(BatchNormalization())  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# add ab extra conv block to JAGG_1
def JAGG_3(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
# dropout of 0.2 - 0.5 is recommended :
# http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Keep in mind dropouts overuse will hurt model performance
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# 7 conv layers total with dropout, padding, maxpooling.  Deepest model yet.
# Unexpectedly, JAGG_4 takes only slightly over 1 hour to complete training.
# JAGG_2 performs slightly better than JAGG_4.
def JAGG_4(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

# JAGG_4 without the ZeroPadding2D
def JAGG_4_2(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))

	return model

def EKAMI(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()
	model.add(BatchNormalization(input_shape=(num_channels, rescaled_dim1, rescaled_dim2)))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.35))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.35))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.35))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.35))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(model_output_layer_size, activation='sigmoid'))        
	return model

# Modified from Source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(num_channels, rescaled_dim1, rescaled_dim2):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224))) # TODO fix dims
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
	model.add(MaxPooling2D((2,2), strides=(2,2)))

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
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	return model