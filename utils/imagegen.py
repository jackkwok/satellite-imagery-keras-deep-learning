import keras as k
from keras.preprocessing.image import ImageDataGenerator

# NormalizedByFeature() offers seemingly no improvement but cost 30 min more to run.
class NormalizedByFeature(object):
    def getTrainGenenerator(self):
		# TODO augment with random rotations for rare classes
		return ImageDataGenerator(
        	featurewise_std_normalization=True,
        	horizontal_flip=True,
        	vertical_flip=True,
        	rescale=None,
        	shear_range=0.0,
        	zoom_range=0.0
        	)
    
    def getValidationGenenerator(self):
		return ImageDataGenerator(
			featurewise_std_normalization=True)
    
    def getTestGenenerator(self):  
 		return ImageDataGenerator(
 			featurewise_std_normalization=True)

    def __repr__(self):
        return 'std_norm h_flip v_flip'

class ScaledDown(object):
    def getTrainGenenerator(self):
      	return ImageDataGenerator(
        	rescale=1./255)

    def getValidationGenenerator(self):
    	return ImageDataGenerator(
    		rescale=1./255)
    
    def getTestGenenerator(self):  
 		return ImageDataGenerator(
 			rescale=1./255)

    def __repr__(self):
        return 'rescale'

class FlipRotateScale(object):
    def getTrainGenenerator(self):
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20.0,
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

    def __repr__(self):
        return 'h_flip v_flip rescale rotate_20'

class GeneralImgGen(object):
    def __init__(self, rescale=1./255, rotation_range=0.0,
        zoom_range=0.0, horizontal_flip=False, vertical_flip=False):
        self.rescale = rescale
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def getTrainGenenerator(self):
        return ImageDataGenerator(
            rescale=self.rescale,
            rotation_range=self.rotation_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            fill_mode='constant', 
            cval=0)

    def getValidationGenenerator(self):
        return ImageDataGenerator(
            rescale=self.rescale)
    
    def getTestGenenerator(self):  
        return ImageDataGenerator(
            rescale=self.rescale)

    def __repr__(self):
        return 'rotation_range:{} horizontal_flip:{} vertical_flip:{}'.format(self.rotation_range, self.horizontal_flip, self.vertical_flip)

        
