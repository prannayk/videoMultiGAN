import tensorflow as tf

def attention_single(X,frames=1,scope):
	batch_size = int(X.shape[0])
	height = int(X.shape[1])
	width = int(X.shape[2])
	embedding_size = int(X.shape[3])
	attention1 = tf.layers.conv2d(X, filters=embedding_size//2, strides=[1,width],
		padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
		reuse=scope.reuse, name="first_layer")
	attention2 = tf.layers.conv2d(attention1, filters=1, strides=[1,width],
		padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
		reuse=scope.reuse, name="first_layer")
	attention_matrix = tf.contrib.layers.softmax(attention2)
	final_layer = tf.reshape(tf.matmul(attention_matrix, X, transpose_a=True),
		shape=[batch_size, height,embedding_size])
	assert frames % height  == 0
	output_layer = final_layer
	for _ in range(frames // height):
		output_layer = tf.concat([output, final_layer], axis=1)
	return output_layer

def attention_multi(X, frames, scope):
	batch_size = int(X.shape[0])
	height = int(X.shape[1])
	width = int(X.shape[2])
	embedding_size = int(X.shape[3])
	attention1 = tf.layers.conv2d(X, filters=embedding_size//2, strides=[1,width],
		padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
		reuse=scope.reuse, name="first_layer_1")
	attention2 = tf.layers.conv2d(attention1, filters=1, strides=[1,width],
		padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
		reuse=scope.reuse, name="first_layer_1")
	attention_matrix = tf.contrib.layers.softmax(attention2)
	final_layer = tf.reshape(tf.matmul(attention_matrix, X, transpose_a=True),
		shape=[batch_size, height,embedding_size])
	output_layer = final_layer
	for i in range(frames // height):
		attention1 = tf.layers.conv2d(X, filters=embedding_size//2, strides=[1,width],
			padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
			reuse=scope.reuse, name="first_layer_%d"%(i+2))
		attention2 = tf.layers.conv2d(attention1, filters=1, strides=[1,width],
			padding='SAME', activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
			reuse=scope.reuse, name="first_layer_%d"%(i+2))
		attention_matrix = tf.contrib.layers.softmax(attention2)
		final_layer = tf.reshape(tf.matmul(attention_matrix, X, transpose_a=True),
			shape=[batch_size, height,embedding_size])
		tf.concat([output_layer,final_layer], axis=1)
	return output_layer

test = tf.placeholder(tf.float32, shape=[100,1,20,300])
with tf.variable_scope("roller") as scope:
	attention_1 = attention_single(test,frames=10,scope)
	attention_2 = attention_multi(test,frames=10, scope)
	print(test.shape)
	print(attention_1.shape)
	print(attention_2.shape)