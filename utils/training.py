from six import string_types
import pandas as pd
import os
import gc

class TrainingSet(object):
	'utility methods for slicing / selecting training samples by criteria'

	def __init__(self, training_path):
		self.labels_df = pd.read_csv(training_path)

		# Build list with unique labels
		label_list = []
		for tag_str in self.labels_df.tags.values:
			labels = tag_str.split(' ')
			for label in labels:
				if label not in label_list:
					label_list.append(label)
					
		# Add one-hot features for every label
		for label in label_list:
			self.labels_df[label] = self.labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

	def one_hot_encoded(self):
		"""One Hot Encode the labels in the training data frame"""
		return self.labels_df

	def samples_with_tags(self, tags, n=None):
		"""Randomly sample n images where each sample has ALL the tags."""
		condition = True
		if isinstance(tags, string_types):
			raise ValueError("argument must be a list of tags, not a single tag.")
		for tag in tags:
			condition = condition & self.labels_df[tag] == 1
		if n is not None:
			#print(condition)
			return self.labels_df[condition].sample(n)
		else:
			return self.labels_df[condition]

	def samples_without_tags(self, tags, n=None):
		"""Randomly sample n images  where each sample does not contain ANY of the tags."""
		condition = True
		if isinstance(tags, string_types):
			raise ValueError("argument must be a list of tags, not a single tag.")
		for tag in tags:
			condition = condition & self.labels_df[tag] == 0
		if n is not None:
			#print(condition)
			return self.labels_df[condition].sample(n)
		else:
			return self.labels_df[condition]