import numpy as np
import numpy.ma as ma
import pandas as pd
import os

from utils.predictor import submission_dataframe

submission_dir = 'D:/Downloads/amazon/my_submissions/'
ensemble_output_dir = submission_dir + 'ensemble/'

submission_files = [
	# WARNING!!!: ONLY USE FILES generated AFTER 6/25/2017 at 7pm (post Kaggle test data patch).
	'submission_densenet121_20170716-214237_score_092702.csv',
	'submission_vgg16_20170706-011852_score_092523.csv',
	#'submission_resnet50_20170703-213500_resnet_score_0913.csv',
	'submission_20170626-025551_score_091200.csv'
]

# Note: Binary Voting is the easiest way to do an ensemble but not always the optimal method.
# Alternatively, average the floating point numbers from 2 predictions before doing thresholding.
def generate_ensemble_submission(ensemble_submission_filename):
	""" generate a submission file based on majority vote amongst the submission files. 
	Ensembling gives at least a +0.002 improvement in leaderboard score. """
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
		predictions = predictions+l
	# average over all submission files
	predictions = predictions/num

	binary_predictions = (np.array(predictions) >= 0.5).astype(int)
	predict_df = pd.DataFrame(binary_predictions, columns = class_names)
	df_submission = pd.read_csv(submission_dir + submission_files[0])
	submit_df = submission_dataframe(df_submission, predict_df)

	ensemble_submission_filepath = ensemble_output_dir + ensemble_submission_filename
	submit_df.to_csv(ensemble_submission_filepath, index=False)
	print('submission file generated: {}'.format(ensemble_submission_filepath))