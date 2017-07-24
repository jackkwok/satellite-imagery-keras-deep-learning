import numpy as np
import numpy.ma as ma
import pandas as pd
import os

from sklearn.metrics import fbeta_score

from utils.predictor import submission_dataframe
from utils.f2thresholdfinder import *

submission_dir = 'D:/Downloads/amazon/my_submissions/'
ensemble_output_dir = submission_dir + 'ensemble/'

# Alternatively, average the floating point numbers from 2 predictions before doing thresholding.
def generate_ensemble_submission(ensemble_submission_filename, submission_files, weights):
	""" 
		Generate a submission file based on majorities votes where votes have different weights.
		Better models should be given higher weights. 
		Weights must add up to 1.0.
		Each submission is weighted according to its performance / confidence.
	"""
	print('ensembling kaggle submission files: {}'.format(submission_files))
	class_names = ['slash_burn', 'clear', 'blooming', 'primary', 'cloudy', 
		'conventional_mine', 'water', 'haze', 'cultivation', 'partly_cloudy', 
		'artisinal_mine', 'habitation', 'bare_ground', 'blow_down', 
		'agriculture', 'road', 'selective_logging']
	num_classes = len(class_names)

	num = len(submission_files)
	for n in range(0, num):
		df = pd.read_csv(submission_dir + submission_files[n])
		for c in class_names:
			df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

		if n == 0:
			names  = df.iloc[:, 0].values
			N = df.shape[0]
			predictions = np.zeros((N, num_classes), dtype=np.float32)

		l = df.iloc[:,2:].values.astype(np.float32)
		predictions = predictions + (l * weights[n])

	binary_predictions = (np.array(predictions) >= 0.5).astype(int)
	predict_df = pd.DataFrame(binary_predictions, columns = class_names)
	df_submission = pd.read_csv(submission_dir + submission_files[0])
	submit_df = submission_dataframe(df_submission, predict_df)

	ensemble_submission_filepath = ensemble_output_dir + ensemble_submission_filename
	submit_df.to_csv(ensemble_submission_filepath, index=False)
	print('submission file generated: {}'.format(ensemble_submission_filepath))

def eval_optimal_ensemble_weights(weights_combos, y_predictions, y_valid, optimizer_func):
	""" 
		Use the validation dataset to figure out the optimal weight distribution.
		
		weights_combos: an array of an array of weights.  Each item represents a single weights combination to test and MUST add up to 1.
		y_predictions: binary predictions
		y_valid: truth binary labels
	
	"""
	# if weights_combos.shape[1] != len(models):
	# 	raise ValueError('length of weights different from number of models!')

	score = 0
	result = None

	for weight_combo in weights_combos:
		curr_score = optimizer_func(weight_combo, y_valid, y_predictions)
		# assumes higher score value is better than lower score
		if curr_score > score:
			score = curr_score
			result = weight_combo
	return result

def weighted_ensemble_f2_score_optimizer(weights_combo, y_valid, y_predictions):
	"""
		An optimizer_func implementation that evaluate F2 scores as the metrics to optimize.
	"""
	y_predict_aggregate = np.zeros((y_valid.shape[0], y_valid.shape[1]), dtype=np.float32)
	for weight, y_predict in zip(weights_combo, y_predictions):
		y_predict_aggregate = y_predict_aggregate + (y_predict.astype(np.float32) * weight)
	binary_predictions = (np.array(y_predict_aggregate) >= 0.5).astype(int)
	f2_score = fbeta_score(y_valid, binary_predictions, beta=2, average='samples')
	print('> F2 score : {} for weights: {}'.format(f2_score, weights_combo))
	return f2_score
