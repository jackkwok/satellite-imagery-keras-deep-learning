import keras as k

from utils.augmentation import *

class CustomImgGenerator(object):
	""" Generate images in batches.  Usage: pass to Keras fit_generator. 
	Perform image augmentations (e.g. flip horizon) 
	these generators will loop indefinitely as required by Keras fit_generator """
	def trainGen(self, x_train, y_train, batch_size):
		i = 0
		limit = x_train.shape[0]
		#print('limit', limit)
		while True:
			x_result = x_train[i: i + batch_size]
			x_result = x_result.transpose(0,2,3,1) # imgaug expects channels last
			x_result = apply_augment_sequence(x_result)
			x_result = x_result / float(255)
			x_result = x_result.transpose(0,3,1,2)

			#print(x_result.shape)

			yield x_result, y_train[i: i + batch_size]
			if i + batch_size > limit:
				i = 0
			else:
				i += batch_size

	def validationGen(self, x_valid, y_valid, batch_size):
		i = 0
		limit = x_valid.shape[0]
		while True:
			yield x_valid[i: i + batch_size] / float(255), y_valid[i: i + batch_size]
			if i + batch_size > limit:
				i = 0
			else:
				i += batch_size

	def testGen(self, x_test, y_test, batch_size):
		i = 0
		limit = x_test.shape[0]
		while True:
			yield x_test[i: i + batch_size] / float(255), y_test[i: i + batch_size]
			if i + batch_size > limit:
				i = 0
			else:
				i += batch_size