from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import load_model
from keras.optimizers import SGD

def resnet50_custom_top_classifier(input_shape, num_classes=17):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(num_classes, activation='sigmoid'))  # softmax replaced with sigmoid for multiclass multlabel classification
	return model

def resnet50_model_custom_top_layers(custom_fc_model_path, num_frozen_layers=25):
	resnet_model = applications.ResNet50(weights='imagenet', include_top=False)
	top_model = load_model(custom_fc_model_path)
	
	# BUG: join the Resnet model to the custom top classifer model somehow
	top_model.input = resnet_model.output # Can't do this
	model = Model(input=resnet_model.input, output=top_model.output)

	for layer in model.layers[num_frozen_layers:]:
		if hasattr(layer, 'trainable'):
			layer.trainable = True

	for layer in model.layers[:num_frozen_layers]:
		if hasattr(layer, 'trainable'):
			layer.trainable = False

	# TODO try Adam because someone has success with Adam
	sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])
	return model
