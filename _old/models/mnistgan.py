import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
import h5py
import numpy
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# sys.path.append( '../util/')
# from loader import *

path = sys.argv[1]
count = int(sys.argv[2])

dictionary =  {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'the': 10, 'digit': 11, 'and': 12, 'are':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'aroun    d':18, 'jumping':19, 'up':20, 'down':21, '.':22, 'is':23, 'left':24, 'right': 25}
rev_dict = zip(dictionary.values(), dictionary.keys())
def loader(path):
	training_data = numpy.ndarray(shape=[50000,64,64,1], dtype=np.int32)
	train_caption_data = list()
	for i in range(count):
		filename="%s/mnist_two_%d_gif.h5"%(path, count)
		f = h5py.File(filename)
		train_set = f['mnist_gif_train'].value
		train_caption = f['mnist_captions_train']
		train_caption = map(lambda x: map(lambda y : en_model[rev_dict[y]],x),train_caption)
		shape = train_set.shape
		train_images = train_set[:,0].resize([shape[0], shape[2], shape[3], shape[1]])
		training_data[count*10000:count*10000 + 10000] = train_images
		train_caption_data += train_caption
	return training_data, np.array(train_caption_data)

training_datam train_caption_data = loader(path)
batch_size = 10
embedding_size = 180
def generator():
	global training_data, train_caption_data, batch_size, embedding_size
	t = np.random.randint(0,training_data-batch_size)
	image_data = training_data[t:t+batch_size]
	caption_data = train_caption_data[t:t+batch_size]
	l = np.random.normal(scale=0.01, size=[batch_size, embedding_size])
	return l, caption_data, image_data


def save_visualization(X, nh_nw, save_path='./mnistimages/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)



class DCGAN():
	def __init__ (self, batch_size, image_shape, embedding_size, 
		num_class=100, dim1 = 1024, dim2 = 128, dim_channel = 1,
		device = '/gpu:0'):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim_channel = dim_channel
		self.device = device
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=1.0/image_size)

	def generate(self, embedding, classes,scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size,1, 1, self.num_class])
			embedding = tf.concat(axis=1, values=[embedding, classes])
			h1 = tf.layers.dense(embedding, units=dim1, activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer = tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layer.l2_regularizer,name='dense_1',
				reuse=scope.reuse)
			h1_concat = tf.concat(axis=1, values=[h1, classes])
			h2 = tf.layers.dense(h1_concat, units=8*8*dim2, activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer = tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layer.l2_regularizer, name='dense_1',
				reuse=scope.reuse)
			h2_concat = tf.concat(axis=3
				[tf.reshape(h2, shape=[self.batch_size,8,8,dim2]), 
				ystack*tf.ones(shape=[self.batch_size, 8, 8, self.num_class])])
			h3 = tf.layers.conv2d_tranpose(inputs=h2_concat, filters = dim3, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer=tf.contrib.layers.l2_regularizer,
				reuse=scope.reuse,name='conv_1')
			h3_concat = tf.concat(axis=3
				[tf.reshape(h2, shape=[self.batch_size,16,16,dim2]), 
				ystack*tf.ones(shape=[self.batch_size, 16, 16, self.num_class])])
			h4 = tf.layers.conv2d_tranpose(inputs=h3_concat, filters = dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer=tf.contrib.layers.l2_regularizer,
				reuse=scope.reuse,name="conv_2")
			h4_concat = tf.concat(axis=3
				[tf.reshape(h2, shape=[self.batch_size,32,32,dim2]), 
				ystack*tf.ones(shape=[self.batch_size, 32, 32, self.num_class])])
			h5 = tf.layers.conv2d_tranpose(inputs=h3_concat, filters = dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer=tf.contrib.layers.l2_regularizer,
				reuse=scope.reuse,name="conv_3")
			return h5

	def discriminate(self, image, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size, 1,1, self.num_class])
			yneed_1 = ystack*tf.ones([self.batch_size, 64, 64, self.num_class])
			yneed_2 = ystack*tf.ones([self.batch_size, 32, 32, self.num_class])
			yneed_3 = ystack*tf.ones([self.batch_size, 16, 16, self.num_class])

			image_proc = tf.norm(tf.concat(axis=3,
				values=[image, yneed_1]), axis=[1,2,3])
			h1 = tf.layers.conv2d(image_proc, filters=dim4, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=tf.contrib.keras.layers.LeakyReLU,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer, 
				activity_regularizer=tf.contrib.layers.l2_regularizer, 
				reuse=scope.reuse,name="conv_1")
			h1_concat = tf.concat(axis=3, values=[h1, yneed_2])
			h2 = tf.layers.conv2d(image_proc, filters=dim3, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=tf.contrib.keras.layers.LeakyReLU,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer, 
				activity_regularizer=tf.contrib.layers.l2_regularizer, 
				reuse=scope.reuse,name="conv_2")
			h2_concat = tf.concat(axis=3, values=[h2, yneed_3])
			h3 = tf.layers.conv2d(image_proc, filters=dim2, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=tf.contrib.keras.layers.LeakyReLU,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layers.l2_regularizer, 
				activity_regularizer=tf.contrib.layers.l2_regularizer, 
				reuse=scope.reuse,name="conv_2")
			h3_reshape = tf.reshape(h3, shape=[-1, 8*8*dim2])
			h3_concat = tf.concat(axis=1, values=[h3_reshape, classes])
			h4 = tf.layers.dense(h3_concat, units=dim1, 
				activation=tf.contrib.keras.layers.LeakyReLU,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer = tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layer.l2_regularizer, name='dense_1',
				reuse=scope.reuse)
			h4_concat = tf.concat(axis=1, values=[h4, classes])
			h4 = tf.layers.dense(h4_concat, units=1, 
				activation=tf.contrib.keras.layers.LeakyReLU,
				kernel_initializer=self.initializer,
				kernel_regularizer=tf.contrib.layers.l2_regularizer,
				activity_regularizer = tf.contrib.layers.l2_regularizer,
				bias_regularizer=tf.contrib.layer.l2_regularizer, name='dense_2',
				reuse=scope.reuse)

	def build_model(self):
		with tf.device(device):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size, self.num_class])
			r_image = tf.placeholder(tf.float32, [self.batch_size, 4096])
			real_image = tf.reshape(r_image, [self.batch_size] + self.image_shape)
			with tf.variable_scope('generator') as scope:
				fake_image = self.generate(embedding, classes, scope)
			g_image = tf.nn.sigmoid(fake_image)
			with tf.variable_scope('discriminator') as scope:
				real_value = self.discriminate(real_image, classes, scope)
			prob_real = tf.nn.sigmoid(real_value)
			with tf.variable_scope('discriminator'):
				scope = tf.get_variable_scope().reuse_variables()
				fake_value = self.discriminate(g_image, classes, scope)
			with tf.variable_scope('generator'):
				scope = tf.get_variable_scope().reuse_variables()
				self.image_samples = self.generate(embedding, classes, scope)
			prob_fake = tf.nn.sigmoid(fake_value)

			d_cost = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
			g_cost = -tf.reduce_mean(tf.log(prob_fake))
			self.placeholders = {
				'embedding' : embedding,
				'classes' : classes,
				'real_image' : r_image
			}
			self.losses = {
				'disc' : d_cost,
				'gen' : g_cost
			}
			self.prob = {
				'disc' : prob_fake,
				'gen' : prob_real	
			}
			with tf.variable_scope('generator') as scope:
				variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
				optimizer_gen = tf.train.AdamOptimizer().minimize(self.losses['gen'], 
					var_list=variables, colocate_gradient_with_ops=True)
			with tf.variable_scope('discriminator') as scope:
				variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
				optimizer_disc = tf.train.AdamOptimizer().minimize(self.losses['disc'],
					var_list=variables, colocate_gradient_with_ops=True)
			self.optimizers = {
				'gen' : optimizer_gen,
				'disc' : optimizer_disc
			}

		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
	def start(self):
		self.session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
		self.init.run()
		return session
	def train(self, generator, epoch, num_examples, sample_input):
		assert num_examples % self.batch_size == 0
		for _ in xrange(epoch):
			for t in range(num_examples // self.batch_size)
			inputs = generator(self.batch_size, self.embedding_size, self.num_classs)
			feed_dict = zip(self.placeholders.values(),inputs)
			_, g_loss_val = session.run([self.optimizers['gen'], self.losses['gen']],feed_dict=feed_dict)
			_, d_loss_val = session.run([self.optimizers['disc'], self.losses['disc']], feed_dict=feed_dict)
			if t%10 == 0 and t > 0:
				print("Done with batches: %d with lossses : %f and %f"%(t*self.batch_size, g_loss_val, d_loss_val))
			print("Saving sample images for reference")
			feed_dict = zip(self.placeholders.values()[:2], sample_input)
			gen_samples = session.run(self.image_sample, feed_dict)
			save_visualization(gen_samples, (32,32),save_path="mnistsamples/sample_%d.jpg"%(ep+1))
			self.saver.save(self.session,name)
			print('Saved session and here we go')

gan = DCGAN(batch_size, [64, 64,1 ], embedding_size, num_class=300)
gan.start()
samples = generator()[:2]
gan.train(generator, 100, 50000, samples)

pritn("Complete GAN code in under 250 lines, done!!")