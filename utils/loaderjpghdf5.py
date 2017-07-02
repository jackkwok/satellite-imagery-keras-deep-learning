import numpy as np
import pandas as pd
import os
import h5py
import cv2
from tqdm import tqdm
from infrared import *
from utils.align import *
from keras.utils.io_utils import HDF5Matrix

# TODO: move these to amazon.cfg config file
data_dir = 'D:/Downloads/amazon/'
cache_dir = data_dir + 'cache/'
train_file_format_jpg = 'train-jpg/{}.jpg'
test_file_format_jpg = 'test/test-jpg/{}.jpg'

training_set_file_path_format = cache_dir + 'train_set_dim{}_rgb_float.h5'
test_set_file_path_format = cache_dir + 'test_set_dim{}_rgb_float.h5'

def subtract_mean(im):
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68

def add_mean(im):
	im[:,:,0] += 103.939
	im[:,:,1] += 116.779
	im[:,:,2] += 123.68


# Load dataset that doesn't fit into RAM.  Use HDF5Matrix.
# We can write our own generator that read data in batches. See detailed discussion:
# https://github.com/fchollet/keras/issues/1627

def load_training_set(df_train, rescaled_dim):
	"""
	Returns HDF5Matrix of the training set.
	Attempts to load data from cache. If data doesnt exist in cache, load from source"""
	training_file_path = training_set_file_path_format.format(rescaled_dim)
	if not os.path.exists(training_file_path):	
		load_training_set_from_source(df_train, rescaled_dim)

	train_x_hdf5 = HDF5Matrix(training_file_path, "training-x")
	train_y_hdf5 = HDF5Matrix(training_file_path, "training-y")

	return train_x_hdf5, train_y_hdf5

def load_training_set_from_source(df_train, rescaled_dim):
	"""Load and return train X and train Y.
	Order of returned samples matches ordering of samples in df_train"""
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
	label_map = {l: i for i, l in enumerate(labels)}
	#x_train_from_src = []
	#y_train_from_src = []

	number_of_images = len(df_train.index)

	training_file_path = training_set_file_path_format.format(rescaled_dim)

	with h5py.File(training_file_path, 'w') as hf:
		x_dataset = hf.create_dataset("training-x", (number_of_images, 3, rescaled_dim, rescaled_dim), dtype='float16')
		y_dataset = hf.create_dataset("training-y", (number_of_images, len(labels)), dtype='i')

		df_index = 0
		
		for f, tags in tqdm(df_train.values, miniters=1000):
			jpg_img_orig = cv2.imread(data_dir + train_file_format_jpg.format(f))
			jpg_img = cv2.resize(jpg_img_orig, (rescaled_dim, rescaled_dim)).astype(np.float16)
			subtract_mean(jpg_img)
			targets = np.zeros(17)
			for t in tags.split(' '):
				targets[label_map[t]] = 1 
			#x_train_from_src.append(jpg_img)
			#y_train_from_src.append(targets)
			x_dataset[df_index] = jpg_img.transpose(2,0,1) # theano ordering
			y_dataset[df_index] = targets
			df_index += 1

# Warning: large data set may not fit in memory (RAM)
def load_test_set(df_test, rescaled_dim):
	"""
	Returns HDF5Matrix of the test set.
	Attempts to load data from cache. If data doesnt exist in cache, load from source"""
	test_set_file_path = test_set_file_path_format.format(rescaled_dim)
	if not os.path.exists(test_set_file_path):
		load_test_set_from_source(df_test, rescaled_dim)
		train_x_hdf5 = HDF5Matrix(test_set_file_path, "test-x")
	
	return train_x_hdf5

def load_test_set_from_source(df_test, rescaled_dim):
	#x_test_from_src = []

	number_of_images = len(df_test.index)

	df_index = 0

	test_set_file_path = test_set_file_path_format.format(rescaled_dim)
	with h5py.File(test_set_file_path, 'w') as hf:
		x_dataset = hf.create_dataset("test-x", (number_of_images, 3, rescaled_dim, rescaled_dim), dtype='float16')
		for f, tags in tqdm(df_test.values, miniters=1000):
			jpg_img_orig = cv2.imread(data_dir + test_file_format_jpg.format(f))
			jpg_img = cv2.resize(jpg_img_orig, (rescaled_dim, rescaled_dim)).astype(np.float16)
			subtract_mean(jpg_img)
			x_dataset[df_index] = jpg_img.transpose(2,0,1) # TODO faster if we batch write to disk instead of one by one.
			df_index += 1
			#x_test_from_src.append(jpg_img)

def load_test_subset_from_cache(rescaled_dim, start, end):
	"""load (from cache) a subset of test data for batch processing"""
	test_set_file_path = test_set_file_path_format.format(rescaled_dim)
	if os.path.exists(test_set_file_path):
		with h5py.File(test_set_file_path, 'r') as hf:
			test_x_local = hf['test-x'][start:end]
	else:
		raise ValueError('data not found in cache')
	return test_x_local

def is_training_set_in_cache(rescaled_dim):
	training_set_file_path = training_set_file_path_format.format(rescaled_dim)
	return os.path.exists(training_set_file_path)

def is_test_set_in_cache(rescaled_dim):
	test_set_file_path = test_set_file_path_format.format(rescaled_dim)
	return os.path.exists(test_set_file_path)

def get_training_set_file_path(rescaled_dim):
	return training_set_file_path_format.format(rescaled_dim)