import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import gc
from tqdm import tqdm

from utils.imagegen import *
from utils.models import *
from utils.loader import *

def submission_dataframe(model, thresholds, data_mask, testset_datagen, rescale_dim, labels, sample_submission_filepath, need_norm_stats):
	# batch_size is limited by amount of RAM in computer.
	batch_size = 1000
	df_submission = pd.read_csv(sample_submission_filepath)
	number_of_samples = df_submission.shape[0]
	test_prediction = predict_in_batches(model, thresholds, data_mask, testset_datagen, number_of_samples, rescale_dim, batch_size, need_norm_stats)
	result = pd.DataFrame(test_prediction, columns = labels)
	preds = []
	for i in tqdm(range(result.shape[0]), miniters=1000):
		a = result.ix[[i]]
		a = a.transpose()
		a = a.loc[a[i] == 1]
		' '.join(list(a.index))
		preds.append(' '.join(list(a.index)))
	df_submission['tags'] = preds
	return df_submission

def predict_in_batches(model, thresholds, data_mask, testset_datagen, number_of_samples, rescale_dim, batch_size, need_norm_stats):
	print('number of test samples:{}'.format(number_of_samples))
	reminder = number_of_samples % batch_size
	number_batches = (number_of_samples - reminder) / batch_size
	print('full batches:{} reminder:{}'.format(number_batches, reminder))

	predict_accumulate = np.array([])

	number_batches = number_batches if (reminder == 0) else (number_batches + 1)

	for i in tqdm(range(0, number_batches)):
		start = i*batch_size
		end = min(start + batch_size, number_of_samples)
		test_subset = load_test_subset_from_cache(rescale_dim, start, end)
		predictions = predict_single_batch(test_subset, model, thresholds, data_mask, testset_datagen, 128, need_norm_stats)
		del test_subset
		predict_accumulate = np.vstack([predict_accumulate, predictions]) if predict_accumulate.size else predictions

	return predict_accumulate

def predict_single_batch(test_subset, model, thresholds, data_mask, testset_datagen, batch_size, need_norm_stats):
	test_subset = test_subset[:, :, :, data_mask]
	#print(test_subset.shape)
	test_subset = test_subset.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
	#print(test_subset.shape)

	if (need_norm_stats):
		# need to compute internal stats like featurewise std and zca whitening
		testset_datagen.fit(test_subset)

	testset_generator = testset_datagen.flow(
		test_subset,
		y=None,
		batch_size=batch_size,
		shuffle=False)

	testset_predict = model.predict_generator(testset_generator, test_subset.shape[0]) # number of test samples
	y_testset_predictions = (np.array(testset_predict) > thresholds).astype(int)
	return y_testset_predictions