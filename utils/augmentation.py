import imgaug as ia
from imgaug import augmenters as iaa

def rotate_batch(images, degrees):
	seq = iaa.Affine(
			rotate=degrees, # rotate by degrees
			order=ia.ALL, # use any of scikit-image's interpolation methods
			cval=0, # if mode is constant
			mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
		)

	return seq.augment_images(images)

def fliplr(images):
	"""Horizontal flip half of all images"""
	seq = iaa.Fliplr(0.5)
	return seq.augment_images(images)

def apply_augment_sequence(images):
	seq = iaa.Sequential(
		[
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5)
			#iaa.Crop(percent=(0, 0.05)), # crop images from each side # no improvement
		],
		random_order=True)
	return seq.augment_images(images)

# Test Time Augmentation (TTA)
def apply_tta_fliplr(images):
	seq = iaa.Fliplr(1.0)
	return seq.augment_images(images)

def apply_tta_flipud(images):
	seq = iaa.Flipud(1.0)
	return seq.augment_images(images)

def apply_tta_flipboth(images):
	seq = iaa.Sequential(
		[
			iaa.Fliplr(1.0),
			iaa.Flipud(1.0)
		],
		random_order=False)
	return seq.augment_images(images)
