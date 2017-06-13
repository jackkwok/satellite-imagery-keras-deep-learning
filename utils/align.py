import numpy as np
import cv2
import matplotlib.pyplot as plt
from spectral import get_rgb
from skimage.measure import structural_similarity as ssim
from skimage.measure import ransac
from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.transform import warp
from skimage.feature import ORB
from skimage.feature import match_descriptors
from skimage.transform import ProjectiveTransform, AffineTransform, SimilarityTransform
from utils.transform import TranslationTransform

from sklearn.preprocessing import MinMaxScaler

class KeypointDetectionException(Exception):
    pass

def match_color_curve_tif2jpg(im_tif, im_jpg):
	# These parameters controls the percentiles used in the match algorithm:
	percentile_eps = 0.5
	function_resoution = 32
	# Lineary distribute the percentiles
	percentiles = np.linspace(percentile_eps, 100-percentile_eps, function_resoution)
	
	# Calculate the percentiles for TIF and JPG, one per channel
	x_per_channel = [np.percentile(im_tif[...,c].ravel(), percentiles) for c in range(3)]
	y_per_channel = [np.percentile(im_jpg[...,c].ravel(), percentiles) for c in range(3)]

	# This is the main part: we use np.interp to convert intermadiate values between
	# percentiles from TIF to JPG
	convert_channel = lambda im, c: np.interp(im[...,c], x_per_channel[c], y_per_channel[c])
	
	# Convert all channels, join and cast to uint8 at range [0, 255]
	tif2jpg = lambda im: np.dstack([convert_channel(im, c) for c in range(3)]).clip(0,255).astype(np.uint8)
	
	# The function could stop here, but we are going to plot a few charts about its results
	return tif2jpg(im_tif[...,:3])

def is_translational(model):
	if model is None:
		return False
	# scale is expected to be near 1.0 and rotation is expected to be very near 0.0.
	if -0.1 < model.rotation < 0.1 and 0.90 < model.scale < 1.1:
		return True
	else:
		return False

def get_matrix(image_tif_bgrn, image_jpg_bgr, verbose=False):
	"""Get similarity transform matrix
	ORB Limitation: https://github.com/scikit-image/scikit-image/issues/1472 """
	im_tif_adjusted = match_color_curve_tif2jpg(image_tif_bgrn, image_jpg_bgr)
	jpg_gray = cv2.cvtColor(image_jpg_bgr,cv2.COLOR_BGR2GRAY).astype(np.uint8)
	tif_gray = cv2.cvtColor(im_tif_adjusted,cv2.COLOR_BGR2GRAY).astype(np.uint8)

	number_of_keypoints = 100

	# Initialize ORB
	# This number of keypoints is large enough for robust results, 
	# but low enough to run quickly. 
	orb = ORB(n_keypoints=number_of_keypoints, fast_threshold=0.05)
	orb2 = ORB(n_keypoints=number_of_keypoints, fast_threshold=0.05)
	try:
		# Detect keypoints
		orb.detect_and_extract(jpg_gray)
		keypoints_jpg = orb.keypoints
		descriptors_jpg = orb.descriptors
		orb2.detect_and_extract(tif_gray)
		keypoints_tif = orb2.keypoints
		descriptors_tif = orb2.descriptors
	except IndexError:
		raise KeypointDetectionException('ORB Keypoint detection failed')

	# Match descriptors between images
	matches = match_descriptors(descriptors_jpg, descriptors_tif, cross_check=True)

	# Select keypoints from
	#   * source (image to be registered)
	#   * target (reference image)
	src = keypoints_jpg[matches[:, 0]][:, ::-1]
	dst = keypoints_tif[matches[:, 1]][:, ::-1]

	model_robust, inliers = ransac((src, dst), 
                                TranslationTransform,
                                min_samples=4, 
                                residual_threshold=1, 
                                max_trials=300)
	if verbose:
		print(inliers)
		print("number of matching keypoints", np.sum(inliers))

	if inliers is None or np.sum(inliers) < 3 or model_robust is None:
		raise ValueError('Possible mismatched JPG and TIF')

	if is_translational(model_robust):
		# we assume src and dst are not rotated relative to each other
		# get rid of any rotational noise introduced during normalization/centering in transform estimate function 
		model_robust.params[0,0] = 1.0
		model_robust.params[1,1] = 1.0
		return model_robust
	else:
		raise ValueError('Invalid Model')

def align_target_tif_to_jpg(image_tif_bgrn, image_jpg_bgr, target, verbose=False):
	"""compute translational tranform matrix mapping image_tif to image_jpg and then apply the same matrix transform to target"""
	warp_matrix = get_matrix(image_tif_bgrn, image_jpg_bgr, verbose)
	warped_target = warp(target, warp_matrix)
	return warped_target
