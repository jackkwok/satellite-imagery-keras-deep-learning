import numpy as np
from keras.metrics import *
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def f2score(truth, predict, label_index):
	"""Get F2 score for a specific label"""
	return fbeta_score(truth[:, label_index], predict[:, label_index], beta=2)
	
def precision_for_label_index(truth, predict, label_index):
	"""Get precision score for a specific label"""
	return precision_score(truth[:, label_index], predict[:, label_index])

def recall_for_label_index(truth, predict, label_index):
	"""Get recall score for a specific label"""
	return recall_score(truth[:, label_index], predict[:, label_index])

# calculate scores per column.
def calculate_stats_for_prediction(truth, predict, decimals=2):
	if truth.shape[0] != predict.shape[0] or truth.shape[1] != predict.shape[1]:
		raise ValueError('shape do not match between truth and predict arguments')

	precision_list = []
	recall_list = []
	f2_score_list = []
	for x in range(0, truth.shape[1]):
		f2_score = f2score(truth, predict, x)
		precision_s = precision_for_label_index(truth, predict, x)
		recall_s = recall_for_label_index(truth, predict, x)
		f2_score_list.append(round(f2_score, decimals))
		precision_list.append(round(precision_s, decimals))
		recall_list.append(round(recall_s, decimals))
	return precision_list, recall_list, f2_score_list

