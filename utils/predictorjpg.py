import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import gc
from tqdm import tqdm

from utils.imagegen import *
from utils.models import *
from utils.loaderjpg import *
from utils.generator import *

# 3 channels only

def make_submission(model, thresholds, rescale_dim, labels, sample_submission_filepath, real_submission_filepath, generator):
	df_submission = pd.read_csv(sample_submission_filepath)
	test_set = load_test_set(df_submission, rescale_dim)
	number_of_samples = df_submission.shape[0]
	probability_prediction_filepath = real_submission_filepath + '.h5'
	predict_df = prediction_dataframe(model, thresholds, labels, test_set, generator, probability_prediction_filepath);
	submit_df = submission_dataframe(df_submission, predict_df)
	submit_df.to_csv(real_submission_filepath, index=False)
	print('submission file generated: {}'.format(real_submission_filepath))

def prediction_dataframe(model, thresholds, labels, test_set, generator, probability_prediction_filepath):
	# batch_size is limited by amount of RAM in computer.  Set smaller batch size for bigger models like ResNet50.
	batch_size = 64
	probability_prediction = predict_probabilities(test_set, model, batch_size, generator)
	# backup the probabilities for easy result ensembling
	np.save(probability_prediction_filepath, probability_prediction)
	test_prediction = predict_binary(probability_prediction, thresholds)
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

def predict_binary(probability_predict, thresholds):
	"""Predict in batches to address limited GPU memory"""
	#print(test_subset.shape)
	
	y_testset_predictions = (np.array(probability_predict) > thresholds).astype(int)
	return y_testset_predictions

def predict_probabilities(test_set, model, batch_size, generator):
	"""Predict in batches to address limited GPU memory"""
	#print(test_subset.shape)
	testset_generator = generator.testGen(test_set, batch_size)
	testset_predict = model.predict_generator(testset_generator, test_set.shape[0]) # number of test samples
	return testset_predict