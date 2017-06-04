import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from spectral import get_rgb
from sklearn.preprocessing import MinMaxScaler

data_dir = 'D:/Downloads/amazon/'
train_file_format_jpg = 'train-jpg/{}.jpg'
test_file_format_jpg = 'test/test-jpg/{}.jpg'
train_file_format_tif = 'train-tif-v2/{}.tif'
test_file_format_tif = 'test/test-tif-v2/{}.tif'

def train_jpg(filename):
	return data_dir + train_file_format_jpg.format(filename)

def train_tif(filename):
	return data_dir + train_file_format_tif.format(filename)

def test_jpg(filename):
	return data_dir + test_file_format_jpg.format(filename)

def test_tif(filename):
	return data_dir + test_file_format_tif.format(filename)
	
def show_jpg(jpg_filename):
	rgb_image_jpg = io.imread(jpg_filename)
	new_style = {'grid': False}
	# visualize JPG
	plt.imshow(cv2.cvtColor(rgb_image_jpg, cv2.COLOR_BGR2RGB))
	plt.title('JPG')
	plt.colorbar()

def show_tiff(tiff_filename):
	bgrn_image = io.imread(tiff_filename)
	new_style = {'grid': False}
	# visualize TIF NIR channel
	plt.imshow(bgrn_image[:,:,3])
	plt.title('NIR')
	plt.colorbar()

def show_tiff_rgb(tiff_filename):
	bgrn_image = io.imread(tiff_filename)
	show_tiff_image_data(bgrn_image)

def show_tiff_image_data(bgrn_image):
	"""Show a rendering of scaled RGB of BGRN 4 channel image matrix"""
	tif_rgb = get_rgb(bgrn_image, [2, 1, 0]) # RGB
	# rescaling to 0-255 range - uint8 for display
	rescaleIMG = np.reshape(tif_rgb, (-1, 1))
	scaler = MinMaxScaler(feature_range=(0, 255))
	rescaleIMG = scaler.fit_transform(rescaleIMG)
	img_scaled = (np.reshape(rescaleIMG, tif_rgb.shape)).astype(np.uint8)
	new_style = {'grid': False}
	plt.imshow(img_scaled)
	plt.title('RGB')
	plt.colorbar()

def show_tiff_nir(bgrn_image):
	# visualize TIF NIR channel
	new_style = {'grid': False}
	plt.imshow(bgrn_image[:,:,3])
	plt.title('NIR')
	plt.colorbar()
