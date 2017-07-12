import tensorflow as tf
import numpy as np
import scipy.misc
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/hdd/hdd/data_backup/prannayk/MNIST_data/", one_hot=True)

class VAEGAN():
	"""docstring for VAEGAN"""
	def __init__(self, batch_size = 16, image_shape= [28,28,3], embedding_size = 128,
		dim1 = 1024, dim2 = 128, dim3 = 64, learning_rate = sys.argv[1:],
		motion_size = 4, num_class_motion=6, num_class_image=13	):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class_image = num_class_image
		self.num_class_motion = num_class_motion
		self.num_class = num_class_image + num_class_motion
		self.zdimension = self.num_class
		self.motion_size = motion_size
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		#self.dim4 = dim4
		self.learning_rate = map(lambda x: float(x), learning_rate)
		# self.learning_rate_1 = floor(learning_rate_1)
		# self.learning_rate_2 = floor(lear/ning_rate_2)
		# assumes square images
		self.lambda_1 = 1
		self.dim_1 = self.image_shape[0]
		self.dim_2 = self.image_shape[0] // 2
		self.dim_4 = self.image_shape[0] // 4
		self.dim_8 = self.image_shape[0] // 8
		self.dim_channel = self.image_shape[-1]
		self.device = "/gpu:0"
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=0.02)

	def learningR(self):
		return self.learning_rate

	def normalize(self, X, reuse=False, name=None, flag=False):
		if not flag : 
			mean , vari = tf.nn.moments(X, 0, keep_dims =True)
		else:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
		return tf.nn.batch_normalization(X, mean, vari, offset=None, 
			scale=None, variance_epsilon=1e-6, name=name)

	def cross_entropy(self, X, flag=True):
		if flag :
			labels = tf.ones_like(X)
		else :
			labels = tf.zeros_like(X)
		softmax = tf.nn.softmax_cross_entropy_with_logits(logits =X, labels=labels)
		return tf.reduce_mean(softmax)

	def discriminate_image(self, image, zvalue, scope):
		with tf.device(self.device):
			ystack = tf.reshape(zvalue, [self.batch_size, 1,1,self.zdimension])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_1 , self.dim_1, self.zdimension])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.zdimension])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.zdimension])
			
			LeakyReLU = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)

			image_proc = tf.concat(axis=3, 
				values=[self.normalize(image, flag=True), yneed_1])
			h1 = tf.layers.conv2d(image_proc, filters=8, kernel_size=[5,5],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = LeakyReLU(h1)
			h1_concat = self.normalize(tf.concat(axis=3, values=[h1_relu, yneed_2]))
			h2 = tf.layers.conv2d(h1_concat, filters=16, kernel_size=[5,5],
				strides=[1,1], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h2_relu = LeakyReLU(h2)
			h2_concat = self.normalize(tf.concat(axis=3, values=[h2_relu, yneed_2]))
			h3 = tf.layers.conv2d(h2_concat, filters=32, kernel_size=[5,5],
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			h3_relu = LeakyReLU(h3)
			h3_concat = self.normalize(tf.concat(axis=3, values=[h3_relu, yneed_3]))
			h4 = tf.layers.conv2d(h3_concat, filters=64, kernel_size=[5,5],
				strides=[1,1], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_4")
			h4_relu = LeakyReLU(h4)
			h4_concat = self.normalize(tf.concat(axis=3, values=[h4_relu, yneed_3]))
			h5 = tf.layers.conv2d(h4_concat, filters=64, kernel_size=[5,5],
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_5")
			h5_relu = LeakyReLU(h3)
			h5_reshape = tf.reshape(h5, shape=[self.batch_size,self.dim_8*self.dim_8*64])
			h5_concat = self.normalize(tf.concat(axis=1, values=[h5_reshape, zvalue]))
			h6 = tf.layers.dense(h5_concat, units=256, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_1',
				reuse=scope.reuse)
			h6_relu = LeakyReLU(h6)
			h6_concat = self.normalize(tf.concat(axis=1, values=[h6_relu, zvalue]))
			h7 = tf.layers.dense(h6_concat, units=self.num_class, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return LeakyReLU(self.normalize(h5, name="last_normalize", reuse=scope.reuse))

	def generate_image(self, embedding, zvalue, scope):
		with tf.device(self.device):
			ystack = tf.reshape(zvalue, shape=[self.batch_size, 1,1 , self.zdimension])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.zdimension])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.zdimension])
			embedding = tf.concat(axis=1, values=[embedding, zvalue])
			h1 = tf.layers.dense(embedding, units=4096, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_reshape = tf.reshape(h1_relu, shape=[self.batch_size, self.dim_4, self.dim_4, 64])
			h1_concat = tf.concat(axis=3, values=[h1_reshape,yneed_1])
			h2 = tf.layers.conv2d_transpose(inputs=h1_concat, filters = 64, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = tf.concat(axis=3, values=[h2_relu, yneed_2])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = 32, 
				kernel_size=[5,5], strides=[1,1], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_2')
			h3_relu = tf.nn.relu(self.normalize(h3))
			h3_concat = tf.concat(axis=3, values=[h3_relu, yneed_2])
			h4 = tf.layers.conv2d_transpose(inputs=h3_concat, filters = self.dim_channel, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_3')
			return tf.nn.sigmoid(h4)

	def encoder_image(self, image, scope):
		with tf.device(self.device):
			LeakyReLU = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
			image_proc = self.normalize(image,flag=True)
			h1 = tf.layers.conv2d(image_proc, filters=32, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = self.normalize(LeakyReLU(h1))
			h2 = tf.layers.conv2d(h1_relu, filters=64, kernel_size=[4,4],
				strides=[1,1], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_2")
			h2_relu = self.normalize(LeakyReLU(h2))
			h3 = tf.layers.conv2d(h2_relu, filters=16, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_3")
			h3_relu = self.normalize(LeakyReLU(h3))
			h3_reshape = tf.reshape(h3_relu, shape=[self.batch_size, self.dim_4*self.dim_4*16])
			h4 = tf.layers.dense(h3_reshape, units=self.embedding_size+self.num_class_image, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return h4 # no activation over last layer of h4 
	def encoder_motion(self, motion_embedding, scope):
		with tf.device(self.device):
			h1 = tf.layers.dense(motion_embedding, units=512,
				activation=None, kernel_initializer=self.initializer,
				name="dense_1", reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h2 = tf.layers.dense(h1, units=self.num_class_motion, 
				activation=None, kernel_initializer=self.initializer,
				name="dense_2", reuse=scope.reuse)
			h2_normalize = self.normalize(h2)
			h3 = tf.nn.softmax(h2)
			return h3

	def discriminate_encode(self, input_embedding, scope):
		h1 = tf.layers.dense(input_embedding, units=750,
			activation=None, kernel_initializer=self.initializer,
			name="dense_1", reuse=scope.reuse)
		h1_relu = tf.nn.relu(self.normalize(h1))
		h2 = tf.layers.dense(h1, units=750,
			activation=None, kernel_initializer=self.initializer, 
			name="dense_2", reuse=scope.reuse)
		h2_relu = tf.nn.relu(self.normalize(h2))
		h3 = tf.layers.dense(h2, units=10, 
			activation=None, kernel_initializer=self.initializer,
			name="dense_3", reuse=scope.reuse)
		return tf.nn.sigmoid(self.normalize(h3))

	def build_model(self):
		image_input = tf.placeholder(tf.float32, shape=[self.batch_size]+ self.image_shape)
		x = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_shape)
		z_s = tf.placeholder(tf.float32, shape=[self.batch_size, self.embedding_size])
		z_c = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class_image])
		image_class_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class_image])
		text_label_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.motion_size])
		z_t = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class_motion])
		placeholders = {
			'image_input' : image_input,
			'x' : x,
			'image_class_input' : image_class_input,
			'text_label_input' : text_label_input,
			'z_s' : z_s,
			'z_c' : z_c,
			'z_t' : z_t
		}
		with tf.variable_scope("encoder") as scope:
			encode = self.encoder_image(image_input, scope)
		with tf.variable_scope("text_encoder") as scope:
			text_encode = self.encoder_motion(text_label_input, scope)
		z_hat_s = encode[:,:self.embedding_size]
		z_hat_c = tf.nn.softmax(encode[:,self.embedding_size:])
		z_hat_t = tf.nn.softmax(text_encode)
		z_hat_input = tf.concat(axis=1, values=[z_hat_c, z_hat_t])
		with tf.variable_scope("generator") as scope:
			x_hat = self.generate_image(z_hat_s,z_hat_input, scope)
			scope.reuse_variables()
			x_dash = self.generate_image(z_s, tf.concat(axis=1, values=[z_c, z_t]),scope)
		with tf.variable_scope("image_discriminator") as scope:
			D_x_hat = self.discriminate_image(x_hat, z_hat_input, scope)
			scope.reuse_variables()
			D_x = self.discriminate_image(x, z_hat_input, scope)
			D_x_dash = self.discriminate_image(x_dash, tf.concat(axis=1, values=[z_c, z_t]),scope)
		with tf.variable_scope("text_classifier") as scope:
			D_z_hat_t = self.discriminate_encode(z_hat_t,scope)
			scope.reuse_variables()
			D_z_t = self.discriminate_encode(z_t, scope)
		with tf.variable_scope("image_classifier") as scope:
			D_z_hat_c = self.discriminate_encode(z_hat_c, scope)
			scope.reuse_variables()
			D_z_c = self.discriminate_encode(z_c, scope)
		losses = dict()
		with tf.variable_scope("losses"):
			losses["reconstruction"] = tf.sqrt(tf.reduce_sum(tf.square(x-x_dash)))
			losses["disc_image_classifier"] = self.cross_entropy(D_z_c, True) + self.cross_entropy(D_z_hat_c,False) + self.cross_entropy(image_class_input, True)
			losses["gen_image_classifier"] = self.cross_entropy(D_z_hat_c, True)
			losses["disc_text_classifier"] = self.cross_entropy(D_z_t,True) + self.cross_entropy(D_z_hat_t, False)
			losses["gen_text_classifier"] = self.cross_entropy(D_z_hat_t, True)
			losses["disc_image_discriminator"] = self.cross_entropy(D_x_hat, False) + self.cross_entropy(D_x_dash, False) + 2*self.cross_entropy(D_x, True)
			losses["generator_image"] = self.cross_entropy(D_x_hat, True) + self.cross_entropy(D_x_dash, True) + (self.lambda_1*losses["reconstruction"])
			losses["encoder"] = losses["gen_image_classifier"] + losses["reconstruction"]
			losses["text_encoder"] = losses["gen_text_classifier"] + self.cross_entropy(D_x_hat, True) + (losses["reconstruction"]*self.lambda_1)
		optimizer = dict()
		with tf.variable_scope("optimizers"):
			optimizer["encoder"] = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5, beta2=0.9).minimize(losses["encoder"])
			optimizer["text_encoder"] = tf.train.AdamOptimizer(self.learning_rate[1], beta1=0.5, beta2=0.9).minimize(losses["text_encoder"])
			optimizer["generator"] = tf.train.AdamOptimizer(self.learning_rate[2],beta1=0.5,beta2=0.9).minimize(losses["generator_image"])
			optimizer["discriminator"] = tf.train.AdamOptimizer(self.learning_rate[3],beta1=0.5, beta2=0.9).minimize(losses["disc_image_discriminator"])
			optimizer["code_discriminator"] = tf.train.AdamOptimizer(self.learning_rate[4],beta1=0.5, beta2=0.9).minimize(losses["disc_image_classifier"])
			optimizer["text_discriminator"] = tf.train.AdamOptimizer(self.learning_rate[5],beta1=0.5, beta2=0.9).minimize(losses["disc_text_classifier"])
		return placeholders, optimizer, losses, x_hat

epoch = 600
batch_size = 64
embedding_size =128
motion_size=4
num_class_image=13
num_class_motion = 5

gan = VAEGAN(batch_size=batch_size, embedding_size=embedding_size, image_shape=[32,32,3], 
	num_class_motion=num_class_motion, num_class_image=num_class_image)

placeholders,optimizers, losses, x_hat = gan.build_model()
session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

saver = tf.train.Saver()

def generate(batch_size):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28,28])
	batch = np.zeros([batch_size, 32, 32,3])
	batch_gen = np.zeros([batch_size, 32, 32,3])
	batch_labels = np.zeros([batch_size, 13])
	batch_labels[:,:10] += batch1_labels
	text_labels = np.zeros([batch_size, 4])
	for i in range(batch_size):
		t = np.random.randint(0,2)
		l = np.random.randint(0,256,[3]).astype(float) / 255
		batch_labels[i,10:] = l
		if t == 0:
			text_labels[i] = np.array([1,1,1,1])
			for j in range(3):
				batch[i,0:28,0:28,j] = batch1[i]*l[j]
				batch_gen[i, 4:32,4:32,j] = batch1[i]*l[j]
		else :
			text_labels[i] = np.array([-1,1,-1,1])
			for j in range(3):
				batch[i,0:28,4:32,j] = batch1[i]*l[j]
				batch_gen[i,4:32,0:28,j] = batch1[i]*l[j]
	return batch, batch_gen, batch_labels, text_labels

def save_visualization(X, nh_nw=(8,8), save_path='../results/%s/sample.jpg'%(sys.argv[4])):
	h,w = X.shape[1], X.shape[2]
	img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

	for n,x in enumerate(X):
		j = n // nh_nw[1]
		i = n % nh_nw[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = x
	np.save("%s.%s"%(save_path.split(".")[0],".npy"), img)
	scipy.misc.imsave(save_path, img)

def random_label(batch_size):
	t = np.random.choice(10, batch_size, replace=True)
	random = np.zeros(shape=[batch_size,13])
	for i in range(batch_size):
		random[i, int(t[i])] = 1
		random[i, 10:] = np.random.randint(0,256,[3]).astype(float) / 255
	return random	

image_sample,image_gen,image_labels, text_labels = generate(64)
save_visualization(image_sample, save_path='../results/vae/image/sample.jpg')
save_visualization(image_gen, save_path='../results/vae/image/sample_gen.jpg')	
saver = tf.train.Saver()
tf.global_variables_initializer().run()

print("Running code: ")

epoch = int(sys.argv[-1])
diter = 5
num_examples = 64000
for ep in range(epoch):
	loss_val = [0,0,0,0,0,0]
	run = 0
	start_time = time.time()
	num_count = 100
	while run < num_examples:
		if ep < 3 or ep % 50 == 0 : 
			iterD = 50
		else :
			iterD = diter
		for t in range(iterD):
			feed_list = generate(batch_size)
			run += batch_size
			feed_dict = {
				placeholders['image_input'] : feed_list[0],
				placeholders['x'] : feed_list[1],
				placeholders['image_class_input'] : feed_list[2],
				placeholders['text_label_input'] : feed_list[3],
				placeholders['z_s'] : np.random.normal(0,1,[batch_size, embedding_size]),
				placeholders['z_c'] : random_label(batch_size),
				placeholders['z_t'] : np.random.normal(0,1,[batch_size, num_class_motion])
			}
			_, loss_val[0] = session.run([optimizers["discriminator"],losses["disc_image_discriminator"]], feed_dict=feed_dict)
			if ep > 10: 
				_, loss_val[1] = session.run([optimizers["code_discriminator"], losses["disc_image_classifier"]], feed_dict=feed_dict)
				_, loss_val[2] = session.run([optimizers["text_discriminator"], losses["disc_text_classifier"]], feed_dict=feed_dict)
				_, loss_val[4] = session.run([optimizers["text_encoder"], losses["text_encoder"]], feed_dict=feed_dict)
			if t % 10 == 0 and t>0:
				print("%d:%d : "%(ep+1,run) + " : ".join(map(lambda x: str(x), loss_val)))
		for t in range(diter):
			feed_list = generate(batch_size)
			run += batch_size
			feed_dict = {
				placeholders['image_input'] : feed_list[0],
				placeholders['x'] : feed_list[1],
				placeholders['image_class_input'] : feed_list[2],
				placeholders['text_label_input'] : feed_list[3],
				placeholders['z_s'] : np.random.normal(0,1,[batch_size, embedding_size]),
				placeholders['z_c'] : random_label(batch_size),
				placeholders['z_t'] : np.random.normal(0,1,[batch_size, num_class_motion])
			}
			if ep > 10 :
				_, loss_val[3] = session.run([optimizers["encoder"], losses["encoder"]], feed_dict=feed_dict)
			_, loss_val[5] = session.run([optimizers["generator"], losses["generator_image"]], feed_dict=feed_dict)
		if run > num_count : 
			num_count = run + 640
			print("%d:%d : "%(ep+1,run) + " : ".join(map(lambda x : str(x),loss_val)) + " " + str(time.time() - start_time))
			start_time = time.time()
	print("DOne with an Epoch")
	images = session.run(x_hat, feed_dict=feed_dict)
	save_visualization(images, save_path="../results/vae/image/sample_%d.jpg"%(ep+1))
