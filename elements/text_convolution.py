# height is 2, that is it

import tensorflow as tf

def convolution_max_pooling(input_layer, filters,scope):

	convolutional_output = tf.layers.conv2d(inputs=input_layer, filter=filters, kernel_size=[2,4],
		strides=[2,1],padding='SAME',activation=tf.tanh, use_bias=False, 
		kernel_initilalizer=tf.random_normal_initializer(stddev=0.02), reuse=scope.reuse,
		name="convolution_text")
	maxpool = tf.layers.max_pooling2d(inputs=convolutional_output, pool_size=[2,4],
		strides=[2,1], padding='SAME', name="maxpool_text")
	return maxpool

with tf.variable_scope("test") as scope:
	layer = tf.placeholder(tf.float32, shape=[100,2,25, 128])
	processed = convolution_max_pooling(layer, 128, scope)
