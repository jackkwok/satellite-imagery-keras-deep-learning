# The purpose is to save a complete history of all trained model files and metadata including scores (accuracy, recall, precision)
# This master record is saved as a csv file and will be used to compare which model is optimal by tying model performance with model files.

# agriculture-20170505-234342.h5 - loss: 0.3806 - acc: 0.8214 - recall: 0.7893 - precision: 0.6662

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

record_file = 'keras_model_master_record.csv'

def record_model_medata(model_file, history, training_time):
	if os.path.exists(record_file):
		df = pd.read_csv(record_file)
	else:
		columns = ['model_file', 'accuracy', 'precision', 'recall', 'training_time']
		df = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

	# get the scores from the last epoch
	accuracy = history.history['acc'][-1]
	precision = history.history['precision'][-1]
	recall = history.history['recall'][-1]

	df = df.append({
		'model_file':model_file, 
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'training_time': training_time}, ignore_index=True)
	#print(df)
	df.to_csv(record_file, index=False)

def test_import():
	print('local module import successful')