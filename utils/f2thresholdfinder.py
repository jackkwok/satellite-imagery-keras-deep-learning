import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from utils.generator import *

def predict_with_optimal_thresholds(x_valid, y_valid, generator, model):
	"""Perform F2 Threshold leaning on validation data"""
	batch_size = 16
	valid_gen = generator.validationGen(x_valid, y_valid, batch_size)
	p_valid = model.predict_generator(valid_gen, x_valid.shape[0])

	optimized_thresholds = f2_optimized_thresholds(y_valid, p_valid)

	y_predictions = (np.array(p_valid) > optimized_thresholds).astype(int)

	precision_s = precision_score(y_valid, y_predictions, average='samples')
	print('>>>> Overall precision score over validation set ' , precision_s)

	recall_s = recall_score(y_valid, y_predictions, average='samples')
	print('>>>> Overall recall score over validation set ' , recall_s)

	# F2 score, which gives twice the weight to recall over precision
	# 'samples' is the evaluation criteria is for the contest
	f2_score = fbeta_score(y_valid, y_predictions, beta=2, average='samples')
	print('>>>> Overall F2 score over validation set ' , f2_score)

	return y_predictions, optimized_thresholds

# WARNING: F2 optimization could decrease performance if test data has different label distributions as train data.
# Code adapted from: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
def f2_optimized_thresholds(y_true, y_pred, verbose=True, resolution=100):
  def score_for_threshold(x):
	averaging_method = 'samples' if (y_true.ndim > 1) else 'binary'
	#p2 = np.zeros_like(p)
	p2 = (y_pred > x).astype(np.int)
	score = fbeta_score(y_true, p2, beta=2, average=averaging_method)
	return score

  classifiers_length = y_true.shape[1] if (y_true.ndim > 1) else 1

  x = [0.5] * classifiers_length
  for i in range(classifiers_length):
	best_i2 = 0
	best_score = 0
	for i2 in range(resolution):
	  i2 /= float(resolution)
	  x[i] = i2
	  score = score_for_threshold(x)
	  if score > best_score:
		best_i2 = i2
		best_score = score
	x[i] = best_i2
	if verbose:
	  print('label:{} threshold:{} score:{}'.format(i, best_i2, best_score))
  return x
