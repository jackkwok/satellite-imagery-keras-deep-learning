import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

def trainHistoryPlot(file_path, history, f2_history, prediction_stats_df):
	"""Generate and save performance plots from history data from model training and validation predictions"""
	fig = plt.figure(figsize=(15, 10))
	subplot0 = fig.add_subplot(231)
	subplot0.plot(f2_history)
	subplot0.set_title('f2 score')
	subplot0.set_ylabel('f2 score')
	subplot0.set_xlabel('epoch')
	subplot0.legend(['val'], loc='upper left')

	# summarize history for recall
	subplot3 = fig.add_subplot(232)
	subplot3.plot(history['recall'])
	subplot3.plot(history['val_recall'])
	subplot3.set_title('recall')
	subplot3.set_ylabel('recall')
	subplot3.set_xlabel('epoch')
	subplot3.legend(['train', 'val'], loc='upper left')

	# summarize history for precision
	subplot2 = fig.add_subplot(233)
	subplot2.plot(history['precision'])
	subplot2.plot(history['val_precision'])
	subplot2.set_title('precision')
	subplot2.set_ylabel('precision')
	subplot2.set_xlabel('epoch')
	subplot2.legend(['train', 'val'], loc='upper left')

	# summarize history for accuracy
	subplot1 = fig.add_subplot(234)
	subplot1.plot(history['acc'])
	subplot1.plot(history['val_acc'])
	subplot1.set_title('accuracy')
	subplot1.set_ylabel('accuracy')
	subplot1.set_xlabel('epoch')
	subplot1.legend(['train', 'val'], loc='upper left')

	# summarize history for loss
	subplot4 = fig.add_subplot(235)
	subplot4.plot(history['loss'])
	subplot4.plot(history['val_loss'])
	subplot4.set_title('model loss')
	subplot4.set_ylabel('loss')
	subplot4.set_xlabel('epoch')
	subplot4.legend(['train', 'val'], loc='upper left')

	# precision and recall for each label
	if (prediction_stats_df):
		subplot5 = fig.add_subplot(236)
		colors = cm.rainbow(np.linspace(0, 1, len(prediction_stats_df['label'])))
		subplot5.scatter(prediction_stats_df['precision'], prediction_stats_df['recall'], c=colors)
		subplot5.set_title('precision & recall')
		subplot5.set_xlabel('precision')
		subplot5.set_ylabel('recall')
		for i, txt in enumerate(prediction_stats_df['label']):
			subplot5.annotate(txt, (prediction_stats_df['precision'][i], prediction_stats_df['recall'][i]))

	fig.savefig(file_path)
	#plt.show()