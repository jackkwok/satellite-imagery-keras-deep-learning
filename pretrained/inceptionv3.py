from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import applications

# Question: does InceptionV3 requires input to be 299x299?

# Dense(2046), 0.25, Dense(1224), 0.25, val_loss: 0.14

def inceptionv3_custom_top_classifier(input_shape, num_classes=17):
	model = Sequential()
	#model.add(AveragePooling2D((8, 8), strides=(8, 8), input_shape=input_shape))
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

def inceptionv3_model_custom_top(num_classes=17, num_frozen_layers=217):
	inceptionv3_model = applications.InceptionV3(weights='imagenet', input_shape=(3, 224, 224), include_top=False)
	
	x = inceptionv3_model.output
	x = Flatten()(x)
	# x = Dense(512, activation='relu')(x)
	# x = Dropout(0.50)(x)
	predictions = Dense(num_classes, activation='sigmoid')(x)
	
	model = Model(input=inceptionv3_model.input, output=predictions)
	model = freeze_layers(model, num_frozen_layers=num_frozen_layers)
	return model

def freeze_layers(model, num_frozen_layers=172):
	"""num_frozen_layers: number of frozen layers total counting from bottom.  0 indexed.
	Default train the top 2 inception blocks, i.e. we will freeze the first 172 layers.
	"""
	for layer in model.layers[num_frozen_layers:]:
		if hasattr(layer, 'trainable'):
			layer.trainable = True

	for layer in model.layers[:num_frozen_layers]:
		if hasattr(layer, 'trainable'):
			layer.trainable = False
	
	optimizer = SGD(lr=0.01, nesterov=True)
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])
	return model