from pretrained.vgg16 import *
from pretrained.vgg19 import *
from pretrained.resnet50 import *
from pretrained.densenet121 import *

def custom_top_model(model_name, num_classes=17, num_frozen_layers=None):
	if model_name == 'vgg16':
		model = vgg16_model_custom_top(num_classes=num_classes, 
								   	   num_frozen_layers=num_frozen_layers)
	elif model_name == 'vgg19':
		model = vgg19_model_custom_top(num_classes=num_classes, 
								   	   num_frozen_layers=num_frozen_layers)
	elif model_name == 'resnet50':
		model = resnet50_model_custom_top(num_classes=num_classes, 
									  	  num_frozen_layers=num_frozen_layers)
	elif model_name == 'densenet121':
		model = densenet121_model_custom_top(num_classes=num_classes, 
											 num_frozen_layers=num_frozen_layers)
	else:
		raise ValueError('Unsupported Model : {}'.format(model_name))
	return model