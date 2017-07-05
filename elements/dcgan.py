import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
import time
import h5py
import numpy
from gensim.models import KeyedVectors
import scipy

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# sys.path.append( '../util/')
# from loader import *
# en_model = KeyedVectors.load_word2vec_format('/media/hdd/hdd/prannayk/temp.vec')


# path = sys.argv[1]
# count = int(sys.argv[2])

# dictionary =  {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'the': 10, 'digit': 11, 'and': 12, 'are':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'around':18, 'jumping':19, 'up':20, 'down':21, '.':22, 'is':23, 'left':24, 'right': 25}
# rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
# def loader(path):
# 	training_data = numpy.ndarray(shape=[50000,64,64,1], dtype=np.int32)
# 	train_caption_data = list()
# 	for i in range(count):
# 		filename="%s/mnist_two_%d_gif.h5"%(path, count)
# 		f = h5py.File(filename)
# 		train_set = f['mnist_gif_train'].value
# 		train_caption = f['mnist_captions_train']
# 		train_caption = map(lambda x: map(lambda y : rev_dict[y],x),train_caption)
# 		train_caption = map(lambda x: map(lambda y: en_model[y], x), train_caption)
# 		shape = train_set.shape
# 		train_images = np.array(map(lambda x: x.reshape(shape[3],shape[4],shape[2]), train_set[:,0]))
# 		training_data[i*10000:i*10000 + 10000] = train_images
# 		train_caption_data += train_caption
# 	return training_data, np.array(train_caption_data)

# training_data, train_caption_data = loader(path)
batch_size = 100
embedding_size = 128
# def generator():
# 	global training_data, train_caption_data, embedding_size
# 	t = np.random.randint(0,len(train_caption_data)-batch_size)
# 	image_shape = reduce(lambda x,y : x*y , training_data[0].shape)
# 	image_data = training_data[t:t+batch_size].reshape(batch_size, image_shape)
# 	caption_data = np.mean(train_caption_data[t:t+batch_size],axis=1)
# 	l = np.random.normal(scale=0.01, size=[batch_size, embedding_size])
# 	return image_data, caption_data, l

### Alternate generator method ###

def generator():
	global batch_size, embedding_size
	image_batch, one_hot_batch = mnist.train.next_batch(batch_size)
	image_batch = image_batch.reshape([batch_size, 28,28,1])
	random_batch = np.random.normal(size=[batch_size, embedding_size])
	return image_batch, one_hot_batch, random_batch


def save_visualization(X, nh_nw, save_path='../mnistimages/sample_0.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)



class DCGAN():
	def __init__ (self, batch_size, image_shape, embedding_size, 
		num_class=10, dim1 = 1024, dim2 = 128, dim3 = 64, dim4=16, dim_channel = 1,
		device = '/gpu:0'):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim4 = dim4
		self.dim_1 = self.image_shape[0]
		self.dim_2 = self.image_shape[0] // 2
		self.dim_4 = self.image_shape[0] // 4
		self.dim_8 = self.image_shape[0] // 8
		self.dim_channel = dim_channel
		self.device = device
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=0.02)

	def normalize(self, X,reuse=False, name=None, flag=False):
		if not flag:
			mean, vari = tf.nn.moments(X, 0, keep_dims=True)
		else:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
		return tf.nn.batch_normalization(X, mean, vari, offset=None, scale=None, variance_epsilon=1e-6)

	def generate(self, embedding, classes,scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size,1, 1, self.num_class])
			embedding = tf.concat(axis=1, values=[embedding, classes])
			h1 = tf.layers.dense(embedding, units=self.dim1, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_concat = tf.concat(axis=1, values=[h1_relu, classes])
			h2 = tf.layers.dense(h1_concat, units=self.dim_4*self.dim_4*self.dim2, 
				activation=tf.tanh, kernel_initializer=self.initializer,
				name='dense_2',	reuse=scope.reuse)
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = self.normalize(tf.concat(axis=3,
				values=[tf.reshape(h2_relu, shape=[self.batch_size,self.dim_4,self.dim_4,self.dim2]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_4, self.dim_4, 
				self.num_class])]),
				flag=True)
#			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = self.dim3, 
#				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
#				kernel_initializer=self.initializer,
#				reuse=scope.reuse,name='conv_1')
#			h3_relu = tf.nn.relu(self.normalize(h3,flag=True))
            #print(h3.get_shape())
#			h3_concat = tf.concat(axis=3,
#				values=[tf.reshape(h3_relu, shape=[self.batch_size,self.dim_4,self.dim_4,self.dim3]), 
#				ystack*tf.ones(shape=[self.batch_size, self.dim_4, self.dim_4, self.num_class])])
			h4 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = self.dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h4_relu = tf.nn.relu(self.normalize(h4,flag=True))
			h4_concat = tf.concat(axis=3,
				values=[tf.reshape(h4_relu, shape=[self.batch_size,self.dim_2,self.dim_2,self.dim4]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_2, self.dim_2, self.num_class])])
			h5 = tf.layers.conv2d_transpose(inputs=h4_concat, filters = self.dim_channel, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			return tf.nn.sigmoid(self.normalize(h5,flag=True))

	def discriminate(self, image, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size, 1,1, self.num_class])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_1, self.dim_1, self.num_class])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.num_class])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.num_class])
		
			LeakyReLU = tf.contrib.keras.layers.LeakyReLU()

			image_proc = self.normalize(tf.concat(axis=3,
				values=[image, yneed_1]),flag=True)
			h1 = tf.layers.conv2d(image_proc, filters=self.dim4, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = LeakyReLU(self.normalize(h1,flag=True))
			h1_concat = tf.concat(axis=3, values=[h1_relu, yneed_2])
			h2 = tf.layers.conv2d(h1_concat, filters=self.dim3, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h2_relu = LeakyReLU(self.normalize(h2, flag=True))
			h2_concat = tf.concat(axis=3, values=[h2_relu, yneed_3])
			h3 = tf.layers.conv2d(h2_concat, filters=self.dim2, kernel_size=[4,4],
				strides=[1,1], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			h3_relu = LeakyReLU(self.normalize(h3,flag=True))
			h3_reshape = tf.reshape(h3_relu, shape=[-1, self.dim_4*self.dim_4*self.dim2])
			h3_concat = self.normalize(tf.concat(axis=1, values=[h3_reshape, classes]),
				name="h3_concat_normalize", reuse=scope.reuse)
			h4 = tf.layers.dense(h3_concat, units=self.dim1, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_1',
				reuse=scope.reuse)
			h4_relu = LeakyReLU(self.normalize(h4))
			h4_concat = self.normalize(tf.concat(axis=1, values=[h4_relu, classes]),
				name="h4_concat_normalize",reuse=scope.reuse)
			h5 = tf.layers.dense(h4_concat, units=1, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return tf.nn.sigmoid(self.normalize(h5,name="last_normalize",reuse=scope.reuse))

	def build_model(self):
		with tf.device(self.device):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size, self.num_class])
			r_image = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1])
			real_image = tf.reshape(r_image, [self.batch_size] + self.image_shape)
			with tf.variable_scope('generator') as scope:
				fake_image = self.generate(embedding, classes, scope)
			g_image = fake_image
			with tf.variable_scope('discriminator') as scope:
				real_value = self.discriminate(real_image, classes, scope)
			with tf.variable_scope('discriminator') as scope:
				scope.reuse_variables()
				fake_value = self.discriminate(g_image, classes, scope)
			with tf.variable_scope('generator') as scope:
				scope.reuse_variables()
				self.image_samples = tf.nn.sigmoid(self.generate(embedding, classes, scope))

			d_cost = -tf.reduce_mean(tf.log(real_value) + (tf.log(1. - fake_value)))
#			d_cost = real_value + fake_value
			g_cost = -tf.reduce_mean(tf.log(fake_value))
#			g_cost = g_image
			self.placeholders = {
				'embedding' : embedding,
				'classes' : classes,
				'real_image' : r_image
			}
			self.losses = {
    			'disc' : d_cost,
				'gen' : g_cost
			}
			#self.prob = {
			#	'disc' : prob_real,
			#	'gen' : prob_fake	
			#}
			with tf.variable_scope('generator') as scope:
				variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
				optimizer_gen = tf.train.AdamOptimizer(3e-2,beta1=0.5).minimize(self.losses['gen'], 
					var_list=variables)
			with tf.variable_scope('discriminator') as scope:
				variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
				optimizer_disc = tf.train.AdamOptimizer(3e-2,beta1=0.5).minimize(self.losses['disc'],
					var_list=variables)
			self.optimizers = {
				'gen' : optimizer_gen,
				'disc' : optimizer_disc
			}
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
	def start(self):
		self.session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
		self.session.run(self.init)
		return self.session
	def train(self, generator, epoch, num_examples, sample_input):
		assert num_examples % self.batch_size == 0
		start = time.time()
		for ep in xrange(epoch):
			start_cycle = time.time()
			average_losses = [0,0]
			for t in range(num_examples // self.batch_size):
				inputs = generator()
				feed_dict = dict(zip(self.placeholders.values(),inputs))
				_, g_loss_val = self.session.run([self.optimizers['gen'], self.losses['gen']],feed_dict=feed_dict)
				#d_loss_val = 0
				_, d_loss_val = self.session.run([self.optimizers['disc'], self.losses['disc']], feed_dict=feed_dict)
				average_losses[0] += g_loss_val
				average_losses[1] += d_loss_val
				if t%100 == 0 and t > 0:
					print("Done with batches: " + str(self.batch_size*t) + " with lossses : " + str(average_losses[0]/100) +  " and " + str(average_losses[1]/100) + " in " + str(time.time() - start))
					average_losses = [0,0]
					start = time.time()
			print("Saving sample images for reference after epoch: %d"%(ep+1))
			feed_dict = dict(zip(self.placeholders.values(), sample_input))
			gen_samples = self.session.run(self.image_samples, feed_dict)
			save_visualization(gen_samples, (10,10), '../mnistimages/sample_output_%d.jpg'%(ep+1))
			# sample_save = gen_samples.reshape([self.batch_size*self.image_shape[0]] + self.image_shape[1:])
			# sample_save = np.concatenate([sample_save, sample_save, sample_save],axis=2)
#			print(np.mean(sample_save[:64]))
#			scipy.misc.imsave("mnistimages/samepl_%d.png"%(ep+1),sample_save)
#			np.save("last_sample.npy", sample_save)
			self.saver.save(self.session, 'model1.ckpt')
			print('Saved session and here we go')
			print("Complete time:" + str(time.time() - start_cycle))
			start_cycle = time.time()

gan = DCGAN(batch_size, [28, 28,1 ], embedding_size, num_class=10)
gan.build_model()
gan.start()
samples = generator()
save_visualization(samples[0], (10,10))
gan.train(generator, 100, 50000, samples)

print("Complete GAN code in under 250 lines, done!!")
