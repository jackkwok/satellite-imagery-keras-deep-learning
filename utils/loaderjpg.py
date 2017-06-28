import numpy as np
import pandas as pd
import os
import h5py
import cv2
from tqdm import tqdm
from infrared import *
from utils.align import *

# TODO: move these to amazon.cfg config file
data_dir = 'D:/Downloads/amazon/'
cache_dir = data_dir + 'cache/'
train_file_format_jpg = 'train-jpg/{}.jpg'
test_file_format_jpg = 'test/test-jpg/{}.jpg'

training_set_file_path_format = cache_dir + 'train_set_dim{}_rgb.h5'
test_set_file_path_format = cache_dir + 'test_set_dim{}_rgb.h5'

# Note: we are loading the entire dataset into memory. Image data will not fit into memory without subsampling.
# We can write our own generator that read data in batches. See detailed discussion:
# https://github.com/fchollet/keras/issues/1627
# Hack to use ImageDataGenerator without giving it all the data in one shot:
# https://stackoverflow.com/questions/44012828/using-imagedatagenerator-with-large-datasets

def load_training_set(df_train, rescaled_dim):
	"""Attempts to load data from cache. If data doesnt exist in cache, load from source"""
	training_file_path = training_set_file_path_format.format(rescaled_dim)
	if os.path.exists(training_file_path):
		with h5py.File(training_file_path, 'r') as hf:
			train_x_local = hf['training-x'][:]
			train_y_local = hf['training-y'][:]
	else:
		train_x_local, train_y_local = load_training_set_from_source(df_train, rescaled_dim)
		with h5py.File(training_file_path, 'w') as hf:
			hf.create_dataset("training-x", data=train_x_local)
			hf.create_dataset("training-y", data=train_y_local)
	return train_x_local, train_y_local

def load_training_set_from_source(df_train, rescaled_dim):
	"""Load and return train X and train Y. Each train X sample has 6 channels (uint8): R, G, B, NDVI, NDWI, NIR in that order.  
	Order of returned samples matches ordering of samples in df_train"""
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
	label_map = {l: i for i, l in enumerate(labels)}
	x_train_from_src = []
	y_train_from_src = []

	for f, tags in tqdm(df_train.values, miniters=1000):
		jpg_img_orig = cv2.imread(data_dir + train_file_format_jpg.format(f))
		jpg_img = cv2.resize(jpg_img_orig, (rescaled_dim, rescaled_dim))
		targets = np.zeros(17)
		for t in tags.split(' '):
			targets[label_map[t]] = 1 
		x_train_from_src.append(jpg_img)
		y_train_from_src.append(targets)
	y_train_from_src = np.array(y_train_from_src, np.uint8) # for GPU compute efficiency
	x_train_from_src = np.array(x_train_from_src, np.uint8)
	print(x_train_from_src.shape)
	print(y_train_from_src.shape)
	return x_train_from_src, y_train_from_src


# Warning: large data set may not fit in memory (RAM)
def load_test_set(df_test, rescaled_dim):
	"""Attempts to load data from cache. If data doesnt exist in cache, load from source"""
	test_set_file_path = test_set_file_path_format.format(rescaled_dim)
	if os.path.exists(test_set_file_path):
		with h5py.File(test_set_file_path, 'r') as hf:
			test_x_local = hf['test-x'][:]
	else:
		test_x_local = load_test_set_from_source(df_test, rescaled_dim)
		with h5py.File(test_set_file_path, 'w') as hf:
			hf.create_dataset("test-x",  data=test_x_local)
	return test_x_local

def load_test_set_from_source(df_test, rescaled_dim):
	x_test_from_src = []
	for f, tags in tqdm(df_test.values, miniters=1000):
		jpg_img_orig = cv2.imread(data_dir + test_file_format_jpg.format(f))
		jpg_img = cv2.resize(jpg_img_orig, (rescaled_dim, rescaled_dim))
		x_test_from_src.append(jpg_img)
	x_test_from_src = np.array(x_test_from_src, np.uint8) # for GPU compute efficiency
	return x_test_from_src

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