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
