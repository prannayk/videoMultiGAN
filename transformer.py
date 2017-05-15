import tensorflow as tf
import numpy as numpy
import scipy.misc as imager

class affine_sequencing():
	"""docstring for affine_sequencing"""
	def __init__(self, batch_size, image_dimension, patch_dimension):
		self.batch_size = batch_size
		self.image_dimension = image_dimension
		self.patch_dimension = patch_dimension

		def patch_to_affine(patch,self):
			weights1 = tf.Variable(tf.random_normal())

