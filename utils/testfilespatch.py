import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

data_dir = 'D:/Downloads/amazon/'

CSVPATH = os.path.join(data_dir, 'test_v2_file_mapping.csv')

TIFPATH = os.path.join(data_dir, 'test/test-tif-v2')
FIXEDPATH = os.path.join(data_dir, 'test/fixed-test-tif')

def copy_and_rename():
	"""https://www.kaggle.com/robinkraft/fix-for-test-jpg-vs-tif-filenames"""
	df = pd.read_csv(CSVPATH)

	if not os.path.exists(FIXEDPATH):
		os.mkdir(FIXEDPATH)

	for index, row in tqdm(df.iterrows(), miniters=1000):
		old = os.path.join(TIFPATH, row['old'])
		new = os.path.join(FIXEDPATH, row['new'])
		shutil.copy(old, new)

copy_and_rename()