import numpy as np
from spectral import get_rgb, ndvi

# geoTiff files store spectral data into 4 channels (B, G, R, NIR) in that order.

# NDVI from the spectral module
def spectral_ndvi(bgrn_input):
    return ndvi(bgrn_input, 2, 3)

# NDVI (vegetations) = (nir-red)/(nir+red)
def compute_ndvi(bgrn_image):
    nrg = get_rgb(bgrn_image, [3, 2, 1]) # NIR-R-G
    ndvi = (nrg[:, :, 0] - nrg[:, :, 1]) / \
        (nrg[:, :, 0] + nrg[:, :, 1]).astype(np.float64)
    return ndvi

# NDWI (water) = (green-nir)/(green+nir)
def compute_ndwi(bgrn_image):
    nrg = get_rgb(bgrn_image, [3, 2, 1]) # NIR-R-G
    ndwi = (nrg[:, :, 2] - nrg[:, :, 0]) / \
        (nrg[:, :, 2] + nrg[:, :, 0]).astype(np.float64)
    return ndwi

# index is in range -1.0 to +1.0 (float).  Normalize it to 0-255 (uint8) for efficient storage along with RGB values
def normalized_index(vi):
	normalized = vi + 1.0 # 0.0 to 2.0
	normalized *= normalized * 255.0/2.0 # 0 to 255
	return np.array(normalized, np.uint8)
