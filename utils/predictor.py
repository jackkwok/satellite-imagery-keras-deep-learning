import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import gc
from tqdm import tqdm

from utils.imagegen import *
from utils.models import *
from utils.loader import *
from utils.generator import *

def make_submission(model, thresholds, data_mask, rescale_dim, labels, sample_submission_filepath, real_submission_filepath):
	df_submission = pd.read_csv(sample_submission_filepath)
	number_of_samples = df_submission.shape[0]
	predict_df = prediction_dataframe(model, thresholds, data_mask, rescale_dim, labels, number_of_samples);
	submit_df = submission_dataframe(df_submission, predict_df)
	submit_df.to_csv(real_submission_filepath, index=False)
	print('submission file generated: {}'.format(real_submission_filepath))

def prediction_dataframe(model, thresholds, data_mask, rescale_dim, labels, number_of_samples):
	# batch_size is limited by amount of RAM in computer.
	batch_size = 1000
	test_prediction = predict_in_batches(model, thresholds, data_mask, number_of_samples, rescale_dim, batch_size)
	result_df = pd.DataFrame(test_prediction, columns = labels)
	return result_df

def submission_dataframe(df_submission, result_dataframe):
	"""Turn a sample submission dataframe into a real prediction result submission dataframe"""
	preds = []
	for i in tqdm(range(result_dataframe.shape[0]), miniters=1000):
		a = result_dataframe.ix[[i]]
		a = a.transpose()
		a = a.loc[a[i] == 1]
		' '.join(list(a.index))
		preds.append(' '.join(list(a.index)))
	df_submission['tags'] = preds
	return df_submission

def predict_in_batches(model, thresholds, data_mask, number_of_samples, rescale_dim, batch_size):
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
		predictions = predict_single_batch(test_subset, model, thresholds, data_mask, 128)
		del test_subset
		predict_accumulate = np.vstack([predict_accumulate, predictions]) if predict_accumulate.size else predictions

	return predict_accumulate

def predict_single_batch(test_subset, model, thresholds, data_mask, batch_size):
	test_subset = test_subset[:, :, :, data_mask]
	#print(test_subset.shape)
	test_subset = test_subset.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
	#print(test_subset.shape)

	gen = CustomImgGenerator()
	testset_generator = gen.testGen(test_subset, batch_size)

	testset_predict = model.predict_generator(testset_generator, test_subset.shape[0]) # number of test samples
	y_testset_predictions = (np.array(testset_predict) > thresholds).astype(int)
	return y_testset_predictions