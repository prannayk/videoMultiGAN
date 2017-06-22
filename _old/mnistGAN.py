import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def weight_variable(shape,stddev):
	x = tf.truncated_normal(shape,stddev=stddev)
	return tf.Variable(x)

def bias_variable(shape):
	x = tf.zeros(shape)
	return tf.Variable(x)

graph = tf.Graph()

batch_size = 50
embedding_size = 120
epochs = 10

def generator(g_input):
	global embedding_size
	weights1 = tf.Variable(tf.random_uniform([embedding_size,128]))
	bias1 = tf.Variable(tf.zeros([128]))
	hidden1 = tf.reshape((tf.matmul(g_input,weights1) + bias1),[-1,4,4,8])
	# first deconvolution layer weights
	weights2 = tf.Variable(tf.random_uniform([3,3,32,8]))
	bias2 = tf.Variable(tf.zeros(shape=([32])))
	# second deconvolution layer weights
	weights3 = tf.Variable(tf.random_uniform([3,3,8,32]))
	bias3 = tf.Variable(tf.zeros(shape=([8])))
	# third deconvolution layer weights
	weights4 = tf.Variable(tf.random_uniform([3,3,1,8]))
	bias4 = tf.Variable(tf.zeros(shape=([1])))
	# layers
	deconv1 = tf.nn.conv2d_transpose(hidden1,filter=weights2,padding='SAME',strides=[1,2,2,1],output_shape=[batch_size,7,7,32])
	hidden2 = tf.nn.relu(deconv1 + bias2)
	deconv2 = tf.nn.conv2d_transpose(hidden2,filter=weights3,padding='SAME',strides=[1,2,2,1],output_shape=[batch_size,14,14,8])
	hidden3 = tf.nn.relu(deconv2 + bias3)
	deconv3 = tf.nn.conv2d_transpose(hidden3,filter=weights4,padding='SAME',strides=[1,2,2,1],output_shape=[batch_size,28,28,1])
	hidden4 = tf.nn.relu(deconv3 + bias4) # this layer is the image
	generator_output = tf.nn.sigmoid(tf.reshape(hidden4, [batch_size,28,28]))
	return generator_output , [weights1,weights2,weights3,weights4,bias1,bias2,bias3,bias4]

def discriminator(train_input,dropout,reuse=False):
	with tf.variable_scope("D",reuse=reuse):
		weight1 = weight_variable([5,5,1,8],1.0)
		bias1 = bias_variable([8])
		train_image = tf.reshape(train_input, [-1,28,28,1])
		hidden1 = tf.nn.relu(tf.nn.conv2d(train_image,weight1,strides=[1,1,1,1],padding='SAME') + bias1)
		hiddenp1 = tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		hiddenp11 = hidden1 / tf.reduce_sum(hidden1)
		# first convolution ^
		# second convolution
		weight2 = weight_variable([5,5,8,32],0.5)
		bias2 = bias_variable([32])
		hidden2 = tf.nn.relu(tf.nn.conv2d(hiddenp11,weight2,strides=[1,1,1,1],padding='SAME') + bias2)
		hiddenp2 = tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		# third convolution
		weight5 = weight_variable([2,2,32,64],0.5)
		bias5 = bias_variable([64])
		hiddenp3 = tf.nn.conv2d(hiddenp2,weight5,strides=[1,1,1,1],padding='SAME') + bias5
		# fully connected layers : 3 (Quite heavy in computation)
		weight3 = weight_variable([7*7*64,1024],1.0)
		bias3 = bias_variable([1024])
		hidden3 = tf.reshape(hiddenp3,[-1,7*7*64])
		hidden4 = tf.matmul(hidden3,weight3) + bias3
		# hidden_drop = tf.nn.dropout(hidden4,dropout)
		hidden_drop = tf.nn.dropout(hidden4 / tf.reduce_sum(hidden4),dropout)
		weight4 = weight_variable([1024,120],1.0)
		weight6 = weight_variable([120,10],1.0)
		bias4 = bias_variable([120])
		bias6 = bias_variable([10])
		hidden5 = tf.matmul(hidden_drop,weight4) + bias4
		hidden6 = tf.matmul(hidden5,weight6) + bias6
		# hidden7 = hidden6 / tf.reduce_sum(hidden6)
		weight7 = weight_variable([10,1],1.0)
		bias7 = bias_variable([1])
		# final output layer
		output = tf.nn.sigmoid(tf.matmul(hidden6,weight7) + bias7)
		return output, [weight1,weight2,weight3,weight4,weight5,weight6,weight7,bias1,bias2,bias3,bias4,bias5,bias6]

# import math
with graph.as_default():
	dropout = tf.placeholder(tf.float32)
	learning_rate = tf.placeholder(tf.float32)
	one = tf.constant(1.0,dtype=tf.float32)
	g_input = tf.placeholder(tf.float32,shape=(batch_size,embedding_size))
	image_input = tf.placeholder(tf.float32,shape=(batch_size,784))
	g_image, g_weights = generator(g_input)
	d_real, d_weight = discriminator(image_input,dropout)
	d_fake, _  = discriminator(g_image,one,reuse=True)
	disc_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
	# disc_loss = -tf.reduce_mean(tf.log(d_real))
	gen_loss = -tf.reduce_mean(tf.log(d_fake))
	# gen_loss = tf.reduce_sum(g_image)

	optimizer = tf.train.AdamOptimizer(1e-4)
	d_train = optimizer.minimize(disc_loss, var_list = d_weight)
	# g_train = optimizer.minimize(disc_loss, var_list = d_weight + g_weights)
	g_train = optimizer.minimize(gen_loss, var_list = g_weights)
	saver = tf.train.Saver()
	init = tf.global_variables_initializer()

num_epochs = 100
average_total_loss = 0
with tf.Session(graph=graph) as session:
	init.run()
	print("Started training")
	for epoch in range(num_epochs):
		for batch in range(mnist.train.num_examples // batch_size):
			if batch % 20 == 0 and batch > 0:
				print(str(disc_loss_val) + " , "+ str(generator_loss))
				print("Training in this Epoch: " + str(batch*batch_size))
			# batch_x, _ = mnist.train.next_batch(batch_size)
			random = np.random.uniform(-1.0,1.0,size=[batch_size,embedding_size])
			# t = mnist.train.next_batch(batch_size)
			# print(len(t))
			feed_dict = {
				dropout : 0.8,
				learning_rate : 1e-4,
				g_input : random,
				image_input : mnist.train.next_batch(batch_size)[0]
			}
			_,disc_loss_val = session.run([d_train,disc_loss],feed_dict=feed_dict)
			_,generator_loss = session.run([g_train,gen_loss],feed_dict=feed_dict)
			average_total_loss += (disc_loss_val + generator_loss)
		# batch_x, _ = mnist.train.next_batch(batch_size)
		random = np.random.uniform(-1.0,1.0,size=[batch_size,embedding_size])
		# t = mnist.train.next_batch(batch_size)
		# print(t.shape)
		feed_dict = {
			dropout : 0.8,
			learning_rate : 1e-4,
			g_input : random,
			image_input : mnist.train.next_batch(batch_size)[0]
		}
		_,generator_loss = session.run([g_train,gen_loss],feed_dict=feed_dict)
		_,disc_loss_val	 = session.run([d_train,disc_loss],feed_dict=feed_dict)
		print("Losses: " + str(generator_loss) + " , " + str(disc_loss_val))
		average_total_loss /= mnist.train.num_examples
		print("Average Loss: " + str(average_total_loss))
		average_total_loss = 0
	print("Ended training, saving state")
	saver.save(session,"mnist_GAN.ckpt")
	print("Saved session")

print("Built graph, Exiting")
exit(0)
