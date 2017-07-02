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

def make_submission(model, thresholds, rescale_dim, labels, sample_submission_filepath, real_submission_filepath):
	df_submission = pd.read_csv(sample_submission_filepath)
	test_set = load_test_set(df, rescale_dim)
	number_of_samples = df_submission.shape[0]
	predict_df = prediction_dataframe(model, thresholds, labels, test_set);
	submit_df = submission_dataframe(df_submission, predict_df)
	submit_df.to_csv(real_submission_filepath, index=False)
	print('submission file generated: {}'.format(real_submission_filepath))

def prediction_dataframe(model, thresholds, labels, test_set):
	# batch_size is limited by amount of RAM in computer.
	batch_size = 1000
	test_prediction = predict_all(test_set, model, thresholds, batch_size)
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

def predict_all(test_subset, model, thresholds, batch_size):
	test_subset = test_subset.transpose(0,3,1,2)  # https://github.com/fchollet/keras/issues/2681
	#print(test_subset.shape)

	gen = CustomImgGenerator()
	testset_generator = gen.testGen(test_subset, batch_size)

	testset_predict = model.predict_generator(testset_generator, test_subset.shape[0]) # number of test samples
	y_testset_predictions = (np.array(testset_predict) > thresholds).astype(int)
	return y_testset_predictions