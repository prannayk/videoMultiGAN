import tensorflow as tf
import numpy as np
import scipy.misc

def batch_normalize(X, eps=1e-6):
	if X.get_shape().ndims == 4 :
		mean = tf.reduce_mean(X,[0,1,2])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2])
		X = (X - mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 2:
		mean = tf.reduce_mean(X,[0])
		stddev = tf.reduce_mean(tf.square(X-mean),[0])
		X = (X - mean)/tf.sqrt(stddev + eps)
	else:
		raise NoImplementationForSuchDimensions
	return X

def lrelu(X, leak = 0.2):
	f1 = (1 + leak)*0.5
	f2 = (1 - leak)*0.5
	return X*f1 + abs(X)*f2

def bce(o,t):
	o  = tf.clip_by_value(o,1e-5,-1e-5)
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o,labels=t))

class VideoGAN():
	def __init__ (self,batch_size = 50,image_shape = [32,32,3],embedding_size = 128,text_embedding = 128,dim1 = 2048, dim2 = 128, dim3 = 64, dim_channel = 3,frames = 16,name="videogan"):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.text_embedding = text_embedding
		self.dim1 = 2048
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim_channel = dim_channel
		self.name = name
		self.frames = frames

		self.dim_4 = image_shape[0] // 4
		self.dim_2 = image_shape[0] // 2
		self.image_input_size = image_shape[0]*image_shape[1]*image_shape[2]

		self.g_weight1 = [tf.Variable(tf.random_normal([embedding_size + text_embedding_size, dim1], stddev = 0.2), name=(self.name+"_generator_weights1_%d"%(i+1))) for i in range(frames)]
		self.g_weight2 = [tf.Variable(tf.random_normal([dim1 + text_embedding_size, dim2*self.dim_4*self.dim_4], stddev = 0.2), name=(self.name+"_generator_weighs2_%d"%(i+1)))  for i in range(frames)]
		self.g_weight3 = [tf.Variable(tf.random_normal([5,5,dim3,dim2+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weights3_%d"%(i+1))) for i in range(frames)]
		self.g_weight4 = [tf.Variable(tf.random_normal([5,5,dim_channel,dim3+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weights4_%d"%(i))) for i in range(frames)]

		self.d_weight1 = tf.Variable(tf.random_normal([5,5,dim_channel+text_embedding_size, dim3],stddev = 0.2), name=(self.name+"_disc_weight1"))
		self.d_weight2 = tf.Variable(tf.random_normal([5,5,dim3+text_embedding_size, dim2],stddev = 0.2), name=(self.random_normalme+"_disc_weight2"))
		self.d_weight3 = tf.Variable(tf.random_normal([dim2*self.dim_4*self.dim_4+text_embedding_size, dim1],stddev = 0.2), name=(self.name+"_disc_weight3"))
		self.d_weight4 = tf.Variable(tf.random_normal([dim1+text_embedding_size,1],stddev = 0.2), name=(self.name+"_disc_weight4"))

	def 
