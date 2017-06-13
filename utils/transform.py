import six
import math
import numpy as np
from scipy import spatial
from scipy import ndimage as ndi
from skimage.transform import *

def _center_and_normalize_points(points):
	"""Center and normalize image points.

	The points are transformed in a two-step procedure that is expressed
	as a transformation matrix. The matrix of the resulting points is usually
	better conditioned than the matrix of the original points.

	Center the image points, such that the new coordinate system has its
	origin at the centroid of the image points.

	Normalize the image points, such that the mean distance from the points
	to the origin of the coordinate system is sqrt(2).

	Parameters
	----------
	points : (N, 2) array
		The coordinates of the image points.

	Returns
	-------
	matrix : (3, 3) array
		The transformation matrix to obtain the new points.
	new_points : (N, 2) array
		The transformed image points.

	"""

	centroid = np.mean(points, axis=0)

	rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])

	norm_factor = math.sqrt(2) / rms

	matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
					   [0, norm_factor, -norm_factor * centroid[1]],
					   [0, 0, 1]])

	pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])

	new_pointsh = np.dot(matrix, pointsh).T

	new_points = new_pointsh[:, :2]
	new_points[:, 0] /= new_pointsh[:, 2]
	new_points[:, 1] /= new_pointsh[:, 2]

	return matrix, new_points

class TranslationTransform(SimilarityTransform):
	"""2D transmation transformation based on similarity transformation of the form:

	..:math:

		X = a0 * x - b0 * y + a1 =
		  = m * x * cos(rotation) - m * y * sin(rotation) + a1

		Y = b0 * x + a0 * y + b1 =
		  = m * x * sin(rotation) + m * y * cos(rotation) + b1

	where ``m`` is a zoom factor and the homogeneous transformation matrix is::

		[[a0  b0  a1]
		 [b0  a0  b1]
		 [0   0    1]]

	Parameters
	----------
	matrix : (3, 3) array, optional
		Homogeneous transformation matrix.
	scale : float, optional
		Scale factor.
	rotation : float, optional
		Rotation angle in counter-clockwise direction as radians.
	translation : (tx, ty) as array, list or tuple, optional
		x, y translation parameters.

	Attributes
	----------
	params : (3, 3) array
		Homogeneous transformation matrix.

	"""

	def __init__(self, matrix=None, translation=None):
		if matrix is not None and translation is not None:
			raise ValueError("You cannot specify the transformation matrix and"
							 " the implicit parameters at the same time.")
		elif matrix is not None:
			if matrix.shape != (3, 3):
				raise ValueError("Invalid shape of transformation matrix.")
			self.params = matrix
		elif translation is None:
			translation = (0, 0)
			rotation = 0.0
			scale = 1.0
			self.params = np.array([
				[math.cos(rotation), - math.sin(rotation), 0],
				[math.sin(rotation),   math.cos(rotation), 0],
				[                 0,                    0, 1]
			])
			self.params[0:2, 0:2] *= scale
			self.params[0:2, 2] = translation
		else:
			# default to an identity transform
			self.params = np.eye(3)

	def estimate(self, src, dst):
		"""Set the transformation matrix with the explicit parameters.

		You can determine the over-, well- and under-determined parameters
		with the total least-squares method.

		Number of source and destination coordinates must match.

		The similarity transformation is defined as::

			X = a0 * x - b0 * y + a1
			Y = b0 * x + a0 * y + b1

		These equations can be transformed to the following form::

			0 = a0 * x - b0 * y + a1 - X
			0 = b0 * x + a0 * y + b1 - Y

		For tranlsation only transformation, b0 = 0 and a0 =1
			0 = a1 - X + x
			0 = b1 - Y + y

		which exist for each set of corresponding points, so we have a set of
		N * 2 equations. The coefficients appear linearly so we can write
		A x = 0, where::

			A   = [[x 1 -y 0 -X]
				   [y 0  x 1 -Y]
					...
					...
				  ]
			x.T = [a0 a1 b0 b1 c3]

		For tranlsation only transformation, b0 = 0 and a0 =1

			A   = [[1 0 -X+x]
				   [0 1 -Y+y]
					...
					...
				  ]
			x.T = [a1 b1 c3]

		In case of total least-squares the solution of this homogeneous system
		of equations is the right singular vector of A which corresponds to the
		smallest singular value normed by the coefficient c3.

		Parameters
		----------
		src : (N, 2) array
			Source coordinates.
		dst : (N, 2) array
			Destination coordinates.

		Returns
		-------
		success : bool
			True, if model estimation succeeds.

		"""
		#print('src.shape', src.shape)
		if src.shape[0] == 0:
			return False

		try:
			src_matrix, src = _center_and_normalize_points(src)
			dst_matrix, dst = _center_and_normalize_points(dst)
		except ZeroDivisionError:
			self.params = np.nan * np.empty((3, 3))
			return False

		xs = src[:, 0]
		ys = src[:, 1]
		xd = dst[:, 0]
		yd = dst[:, 1]
		rows = src.shape[0]

		A = np.zeros((rows * 2, 3))
		A[:rows, 0] = 1
		A[rows:, 1] = 1
		A[:rows, 2] = xd - xs
		A[rows:, 2] = yd - ys

		_, _, V = np.linalg.svd(A)

		#print('V shape', V.shape) # ('V shape', (4L, 4L))

		# solution is right singular vector that corresponds to smallest
		# singular value
		a1, b1 = - V[-1, :-1] / V[-1, -1]
		b0 = 0
		a0 = 1

		S = np.array([[a0, -b0, a1],
					  [b0,  a0, b1],
					  [ 0,   0,  1]])

		#print('before', S)
		# De-center and de-normalize
		S = np.dot(np.linalg.inv(dst_matrix), np.dot(S, src_matrix))

		#S[0,0] = 1
		#S[1,1] = 1
		#S[0,1] = 0
		#S[1,0] = 0

		#print('after', S)

		self.params = S

		return True

	@property
	def scale(self):
		if abs(math.cos(self.rotation)) < np.spacing(1):
			# sin(self.rotation) == 1
			scale = self.params[1, 0]
		else:
			scale = self.params[0, 0] / math.cos(self.rotation)
		return scale

	@property
	def rotation(self):
		return math.atan2(self.params[1, 0], self.params[1, 1])

	@property
	def translation(self):
		return self.params[0:2, 2]