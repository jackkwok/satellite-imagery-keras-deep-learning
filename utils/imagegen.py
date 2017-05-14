import keras as k
from keras.preprocessing.image import ImageDataGenerator

class NormalizedByFeature(object):
    def getTrainGenenerator(self):
		# TODO augment with random rotations for rare classes
		return ImageDataGenerator(
        	featurewise_std_normalization=True,
        	rescale=None,
        	shear_range=0.0,
        	zoom_range=0.0,
        	horizontal_flip=True,
        	vertical_flip=True)
    
    def getValidationGenenerator(self):
		return ImageDataGenerator(
			featurewise_std_normalization=True)
    def getTestGenenerator(self):  
 		return ImageDataGenerator(
 			featurewise_std_normalization=True)

class ScaledDown(object):
    def getTrainGenenerator(self):
      	return ImageDataGenerator(
        	rescale=1./255,
        	shear_range=0.0,
        	zoom_range=0.0,
        	horizontal_flip=True,
        	vertical_flip=True)

    def getValidationGenenerator(self):
    	return ImageDataGenerator(
    		rescale=1./255)
    
    def getTestGenenerator(self):  
 		return ImageDataGenerator(
 			rescale=1./255)

def getGenenerator(generator_type):
    generator_type.getGenenerator()
 
 
