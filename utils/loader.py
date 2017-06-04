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
train_file_format_tif = 'train-tif-v2/{}.tif'
test_file_format_tif = 'test/test-tif-v2/{}.tif'
training_set_file_path_format = cache_dir + 'train_set_dim{}_rgb_ndvi_ndwi_nir_align.h5'
test_set_file_path_format = cache_dir + 'test_set_dim{}_rgb_ndvi_ndwi_nir_align.h5'

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
	"""load and return train X and train Y. Each train X sample has 6 channels (uint8): R, G, B, NDVI, NDWI, NIR in that order.  Order of returned samples matches ordering of samples in df_train"""
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
	label_map = {l: i for i, l in enumerate(labels)}
	x_train_from_src = []
	y_train_from_src = []

	for f, tags in tqdm(df_train.values, miniters=1000):
		img = cv2.imread(data_dir + train_file_format_jpg.format(f))
		bgrn_img = cv2.imread(data_dir + train_file_format_tif.format(f), cv2.IMREAD_UNCHANGED)
		img = cv2.resize(img, (rescaled_dim, rescaled_dim))
		bgrn_img =cv2.resize(bgrn_img, (rescaled_dim, rescaled_dim))
		combined_img = combine_derived_channels(img, bgrn_img, f)

		targets = np.zeros(17)
		for t in tags.split(' '):
			targets[label_map[t]] = 1 
		x_train_from_src.append(combined_img)
		y_train_from_src.append(targets)
	y_train_from_src = np.array(y_train_from_src, np.uint8) # for GPU compute efficiency
	x_train_from_src = np.array(x_train_from_src, np.uint8)
	print(x_train_from_src.shape)
	print(y_train_from_src.shape)
	return x_train_from_src, y_train_from_src

def combine_derived_channels(jpg_img, bgrn_img, f):
	"""Stack R,G,B from JPG and derived NDVI, NDWI, and NIR from TIFF into a 6 channel image matrix"""
	ndvi = normalized_index(spectral_ndvi(bgrn_img))
	ndvi = np.expand_dims(ndvi, axis=2)

	ndwi = normalized_index(compute_ndwi(bgrn_img))
	ndwi = np.expand_dims(ndwi, axis=2)

	nir = bgrn_img[:, :, 3]
	nir = np.expand_dims(nir, axis=2)

	# combine the TIFF derived values before realignment in one pass
	tiff_derived = np.concatenate((ndvi, ndwi, nir), axis=2)

	# realignment to address TIFF-JPG misalignment in data sets
	try:
		tiff_derived_aligned = align_target_tif_to_jpg(bgrn_img, jpg_img, tiff_derived, verbose=True)
		# combine the RGB values from jpg, NDVI, NDWI, and NIR value into one array
		combined_img = np.concatenate((jpg_img, tiff_derived_aligned), axis=2)
	except cv2.error:
		print('Alignment Error: TIF and JPG not correlated {}'.format(f))
		# tiff_derived_aligned = tiff_derived 
		# hunch: better to fill with all zeros than using garbage data?
		tiff_derived.fill(0)
		combined_img = np.concatenate((jpg_img, tiff_derived), axis=2)
	
	return combined_img

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
		img = cv2.imread(data_dir + test_file_format_jpg.format(f))
		bgrn_img = cv2.imread(data_dir + test_file_format_tif.format(f), cv2.IMREAD_UNCHANGED)
		img = cv2.resize(img, (rescaled_dim, rescaled_dim))
		bgrn_img =cv2.resize(bgrn_img, (rescaled_dim, rescaled_dim))
		combined_img = combine_derived_channels(img, bgrn_img, f)

		x_test_from_src.append(combined_img)
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