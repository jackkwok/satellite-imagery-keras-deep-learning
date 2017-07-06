from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam

def resnet50_custom_top_classifier(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# val_loss: 0.0988
# experimental hypothesis: add a new dense layer should improve val_loss 
def resnet50_custom_top_classifier_experimental(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))  #2048
	model.add(Dense(256, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# worse than 1
def resnet50_custom_top_classifier_experimental_2(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# worse than 1
def resnet50_custom_top_classifier_experimental_3(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# worse than 1
def resnet50_custom_top_classifier_experimental_4(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# DONE: update resnet50_model_custom_top to use the Best top classifier
# 
# dense: 256 dropout: 0.50 val_loss: 0.0978
# dense: 256 dropout: 0.75 val_loss: 0.0984
# dense: 128 dropout: 0.50 val_loss: 0.0988
# dense: 128 dropout: 0.25 val_loss: 0.0975
# dense: 256 dropout: 0.25 val_loss: 0.0982
# dense: 512 dropout: 0.25 val_loss: 0.0977
# dense: 512 dropout: 0.50 val_loss: 0.0970 [Best] 
def resnet50_custom_top_classifier_experimental_5(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))  #2048
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

# 512, 0.5, 512, 0.5 val_loss: 0.0978
# 1024, 0.5, 512, 0.5 val_loss: 0.0999
# 512, 0.5, 256, 0.5 val_loss: 0.0982
def resnet50_custom_top_classifier_experimental_6(input_shape, num_classes=17):
	"""Warning: there seems to be no way to load weights trained from this model into our modified Resnet50 model."""
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

def resnet50_model_custom_top(num_classes=17, num_frozen_layers=175):
	# [7/5/17] Custom top updated to the optimal top model found in previous bottleneck experiments.
	# ResNet50 input_shape: optional shape tuple, only to be specified if include_top is False 
	# (otherwise the input shape has to be  (224, 224, 3) (with channels_last data format) 
	# or (3, 224, 224) (with channels_first data format). It should have exactly 3 inputs channels, 
	# and width and height should be no smaller than 197. E.g. (200, 200, 3) would be one valid value.
	resnet_model = applications.ResNet50(weights='imagenet', input_shape=(3, 224, 224), include_top=False)
	
	x = resnet_model.output
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.50)(x)
	predictions = Dense(num_classes, activation='sigmoid')(x)
	
	model = Model(input=resnet_model.input, output=predictions)
	model = freeze_layers(model, num_frozen_layers=num_frozen_layers)
	return model

def freeze_layers(model, num_frozen_layers=175):
	"""num_frozen_layers: number of frozen layers total counting from bottom.  0 indexed."""
	for layer in model.layers[num_frozen_layers:]:
		if hasattr(layer, 'trainable'):
			layer.trainable = True

	for layer in model.layers[:num_frozen_layers]:
		if hasattr(layer, 'trainable'):
			layer.trainable = False
	
	# use Adam for top classify layer training because we know it works well
	if (num_frozen_layers == 175):
		optimizer = Adam(lr=0.001)
	else:
		# TODO try Adam because someone has success with Adam
		optimizer = Adam(lr=0.0001)
		#optimizer = SGD(lr=0.001, nesterov=True)
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])
	return model
