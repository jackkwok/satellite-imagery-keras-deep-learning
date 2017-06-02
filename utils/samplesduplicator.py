import numpy as np
import pandas as pd
import numpy.ma as ma

def duplicate_train_samples(train_X, train_Y, dup_target_y, multiplier=2):
	"""duplicate samples.  train_Y has N 1,0 tuples.  dup_target_y in [0,N). multiplier must be >=2. append to end of input arrays"""
	
	target_y = train_Y[:,dup_target_y]
	#print(target_y)
	mask = ma.make_mask(target_y)
	additional_X = train_X[mask]
	#print(additional_X)
	additional_Y = train_Y[mask]
	#print(additional_Y)
	
	for i in range(1, multiplier):
		train_X = np.concatenate((train_X, additional_X), axis=0)
		train_Y = np.concatenate((train_Y, additional_Y), axis=0)

	return train_X, train_Y

# def duplicate_samples(train_X, train_Y, indexes_to_duplicate, multiplier=2):
# 	"""duplicate samples with indexes. append to end of input arrays"""
# 	additional_X = train_X[indexes_to_duplicate]
# 	additional_Y = train_Y[indexes_to_duplicate]

# 	for i in range(1, multiplier):
# 		train_X = np.concatenate((train_X, additional_X), axis=0)
# 		train_Y = np.concatenate((train_Y, additional_Y), axis=0)

# 	return train_X, train_Y
