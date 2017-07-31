import tensorflow as tf
import numpy as np
import scipy.misc
import sys
from generator import rot_text_generator as generate
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/hdd/hdd/data_backup/prannayk/MNIST_data/", one_hot=True)

total_size = 64000
batch_size = 16

class VAEGAN():
	"""docstring for VAEGAN"""
	def __init__(self, batch_size = 16, image_shape= [28,28,3], embedding_size = 128,
			learning_rate = sys.argv[1:], motion_size = 5, num_class_motion=12, 
			num_class_image=13, frames=2, frames_input=3, total_size = 64000, video_create=False):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.image_input_shape = list(image_shape)
		self.image_create_shape = list(image_shape)
		self.frames_input = frames_input
		self.frames = frames
		self.image_input_shape[-1] *= self.frames_input
		self.image_create_shape[-1] *= self.frames
		self.num_class_image = num_class_image
		self.num_class_motion = num_class_motion
		self.num_class = num_class_image
		self.embedding_size = embedding_size 
		self.zdimension = self.num_class
		self.motion_size = motion_size
		self.learning_rate = map(lambda x: float(x), learning_rate)
		self.lambda_1 = 10
		self.dim_1 = self.image_shape[0]
		self.dim_2 = self.image_shape[0] // 2
		self.dim_4 = self.image_shape[0] // 4
		self.dim_8 = self.image_shape[0] // 8
		self.dim_16 = self.image_shape[0] // 16
		self.dim_channel = self.image_shape[-1]
		self.device = "/gpu:0"
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=0.02)
		self.first_time = True
		self.total_size = total_size
		self.batch_size = batch_size
		self.create_dataset()
		self.video_create = video_create
	def learningR(self):
		return self.learning_rate

	def normalize(self, X, reuse=False, name=None, flag=False):
		if not flag : 
			mean , vari = tf.nn.moments(X, 0, keep_dims =True)
		else:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
		return tf.nn.batch_normalization(X, mean, vari, offset=None, 
			scale=None, variance_epsilon=1e-6, name=name)

	def gan_loss(self, X, flag=True):
		if flag :
			softmax = tf.log(X)
		else :
			softmax = tf.log(1 - X)
		return -tf.reduce_mean(softmax)

	def create_dataset(self):
		total_size = self.total_size
		batch_size = self.batch_size
		image_start = np.zeros(shape=[total_size] + self.image_input_shape)
		image_gen = np.zeros(shape=[total_size] + self.image_create_shape)
		image_labels = np.zeros(shape=[total_size, 13])
		image_motion_labels = np.zeros(shape=[total_size, 5])
		for i in range(total_size // batch_size):
			if i % 1000 == 0:
				print(i)
			output_list = generate(batch_size, self.frames)
			image_start[i*batch_size : i*batch_size + batch_size] = output_list[0]
			image_gen[i*batch_size:i*batch_size + batch_size] = output_list[1]
			image_labels[i*batch_size:i*batch_size + batch_size] = output_list[2]
			image_motion_labels[i*batch_size:i*batch_size + batch_size] = output_list[3]
		dataset = {
			"image_start" : image_start,
			"image_gen" : image_gen,
			"image_labels" : image_labels,
			"image_motion_labels" : image_motion_labels
		}
		self.dataset = dataset
		self.iter=0
	def generate_batch(self):
		list_output= []
		list_output.append(self.dataset["image_start"][self.iter:self.iter + self.batch_size])
		list_output.append(self.dataset["image_gen"][self.iter:self.iter + self.batch_size])
		list_output.append(self.dataset["image_labels"][self.iter:self.iter + self.batch_size])
		list_output.append(self.dataset["image_motion_labels"][self.iter:self.iter + self.batch_size])
		self.iter = (self.iter + self.batch_size) % self.total_size
		return list_output
	def discriminate_image(self, image, zvalue, scope):
		with tf.device(self.device):
			ystack = tf.reshape(zvalue, [self.batch_size, 1,1,self.zdimension])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_1 , self.dim_1, self.zdimension])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.zdimension])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.zdimension])
			yneed_4 = ystack*tf.ones([self.batch_size, self.dim_8, self.dim_8, self.zdimension])
			
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
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_4")
			h4_relu = LeakyReLU(h4)
			h4_concat = self.normalize(tf.concat(axis=3, values=[h4_relu, yneed_4]))
			h5 = tf.layers.conv2d(h4_concat, filters=64, kernel_size=[5,5],
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_5")
			h5_relu = LeakyReLU(h3)
			h5_reshape = tf.reshape(h5, shape=[self.batch_size,self.dim_16*self.dim_16*64])
			h5_concat = self.normalize(tf.concat(axis=1, values=[h5_reshape, zvalue]))
			h6 = tf.layers.dense(h5_concat, units=256, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_1',
				reuse=scope.reuse)
			h6_relu = LeakyReLU(h6)
			h6_concat = self.normalize(tf.concat(axis=1, values=[h6_relu, zvalue]))
			h7 = tf.layers.dense(h6_concat, units=1, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return tf.nn.sigmoid(LeakyReLU(self.normalize(h5, name="last_normalize", reuse=scope.reuse)))
	def generate_image(self, embedding, zvalue, scope):
		with tf.device(self.device):
			ystack = tf.reshape(zvalue, shape=[self.batch_size, 1,1 , self.zdimension])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.zdimension])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.zdimension])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_8, self.dim_8, self.zdimension])
			embedding = tf.concat(axis=1, values=[embedding, zvalue])
			h1 = tf.layers.dense(embedding, units=4096, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_reshape = tf.reshape(h1_relu, shape=[self.batch_size, self.dim_8, self.dim_8, 64])
			h1_concat = tf.concat(axis=3, values=[h1_reshape,yneed_3])
			h2 = tf.layers.conv2d_transpose(inputs=h1_concat, filters = 64, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = tf.concat(axis=3, values=[h2_relu, yneed_1])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = 32, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
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
			h1 = tf.layers.conv2d(image_proc, filters=48, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = self.normalize(LeakyReLU(h1))
			h2 = tf.layers.conv2d(h1_relu, filters=64, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
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
			h3_reshape = tf.reshape(h3_relu, shape=[self.batch_size, self.dim_8*self.dim_8*16])
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
		h3 = tf.layers.dense(h2, units=1, 
			activation=None, kernel_initializer=self.initializer,
			name="dense_3", reuse=scope.reuse)
		return tf.nn.sigmoid(self.normalize(h3))
	def create_frames(self, image_input, x, z_s, z_c, image_class_input, text_label_input, z_t):
		with tf.variable_scope("encoder") as scope:
			if not self.first_time :
				scope.reuse_variables()
			encode = self.encoder_image(image_input, scope)
		with tf.variable_scope("text_encoder") as scope:
			if not self.first_time :
				scope.reuse_variables()
			text_encode = self.encoder_motion(text_label_input, scope)
		z_hat_s = encode[:,:self.embedding_size]
		z_hat_c = tf.nn.softmax(encode[:,self.embedding_size:])
		z_hat_t = tf.nn.softmax(text_encode)
		z_hat_input = tf.concat(axis=1, values=[z_hat_s, z_hat_t])
		with tf.variable_scope("generator") as scope:
			if not self.first_time :
				scope.reuse_variables()
			x_hat = self.generate_image(z_hat_input, z_hat_c, scope)
			scope.reuse_variables()
			x_dash = self.generate_image(tf.concat(axis=1, values=[z_s, z_t]),z_c,scope)
			x_gen = self.generate_image(z_hat_input,z_hat_c, scope)
		with tf.variable_scope("image_discriminator") as scope:
			if not self.first_time :
				scope.reuse_variables()
			D_x_hat = self.discriminate_image(x_hat, z_hat_c, scope)
			scope.reuse_variables()
			D_x = self.discriminate_image(x, image_class_input, scope)
			D_x_dash = self.discriminate_image(x_dash, z_c,scope)
			D_x_gen = self.discriminate_image(x_gen, z_hat_c, scope)
		with tf.variable_scope("text_classifier") as scope:
			if not self.first_time :
				scope.reuse_variables()
			D_z_hat_t = self.discriminate_encode(z_hat_t,scope)
			scope.reuse_variables()
			D_z_t = self.discriminate_encode(z_t, scope)
		with tf.variable_scope("image_classifier") as scope:
			if not self.first_time :
				scope.reuse_variables()
			D_z_hat_c = self.discriminate_encode(z_hat_c, scope)
			scope.reuse_variables()
			D_z_c = self.discriminate_encode(z_c, scope)
			D_z_real = self.discriminate_encode(image_class_input, scope)
		with tf.variable_scope("style_classifier") as scope:
			if not self.first_time :
				scope.reuse_variables()
			D_z_hat_s = self.discriminate_encode(z_hat_s, scope)
			scope.reuse_variables()
			D_z_s = self.discriminate_encode(z_s, scope)
		self.first_time = False
		return x_hat, x_gen, D_x_hat, D_x, D_x_dash, D_x_gen, D_z_hat_c, D_z_c, D_z_real, D_z_hat_s, D_z_s, D_z_hat_t, D_z_t
	def build_model(self):
		image_input = tf.placeholder(tf.float32, shape=[self.batch_size]+ self.image_input_shape)
		x = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_create_shape)
		z_s = tf.placeholder(tf.float32, shape=[self.batch_size*self.frames, self.embedding_size])
		z_c = tf.placeholder(tf.float32, shape=[self.batch_size*self.frames, self.num_class_image])
		image_class_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class_image])
		text_label_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.motion_size])
		z_t = tf.placeholder(tf.float32, shape=[self.batch_size*self.frames, self.num_class_motion])
		placeholders = {
			'image_input' : image_input,
			'x' : x,
			'image_class_input' : image_class_input,
			'text_label_input' : text_label_input,
			'z_s' : z_s,
			'z_c' : z_c,
			'z_t' : z_t
		}
		list_values = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
		next_image_input = image_input
		for i in range(self.frames):
			print(i) 
			list_return = self.create_frames(next_image_input, x[:,:,:,3*i:3*i+3], 
				z_s[self.batch_size*i:self.batch_size*i+self.batch_size], z_c[self.batch_size*i:self.batch_size*i+self.batch_size], 
				image_class_input, text_label_input, z_t[self.batch_size*i:self.batch_size*i+self.batch_size])
			for count,item in enumerate(list_return):
				list_values[count].append(item)
			if self.video_create :
				next_image_input = tf.concat(axis=3, values=[next_image_input[:,:,:,3:], x[:,:,:,3*i:3*i+3]])
			else :
				next_image_input = tf.concat(axis=3, values=[next_image_input[:,:,:,3:], list_output[0]])

		# second_image_input = tf.concat(axis=3, values=[image_input[:,:,:,3:],x_1_hat])
		# x_2_hat, x_2_gen, D_2_x_hat, D_2_x, D_2_x_dash, D_2_x_gen, D_2_z_hat_c, D_2_z_c, D_2_z_real, D_2_z_hat_s, D_2_z_s,  D_2_z_hat_t, D_2_z_t = self.create_frames(second_image_input, x[:,:,:,3:], 
			# z_s[self.batch_size:], z_c[self.batch_size:], image_class_input, text_label_input, z_t[self.batch_size:])
		x_hat = tf.concat(axis=3, values=list_values[0])
		print("Completed concat")
		x_gen = tf.concat(axis=3, values=list_values[1])
		print("Completed concat")
		D_x_hat = tf.concat(axis=0, values=list_values[2])
		print("Completed concat")
		D_x_gen = tf.concat(axis=0, values=list_values[5])
		print("Completed concat")
		D_x = tf.concat(axis=0, values=list_values[3])
		print("Completed concat")
		D_x_dash = tf.concat(axis=0, values=list_values[4])
		print("Completed concat")
		D_z_hat_c = tf.concat(axis=0, values=list_values[6])
		print("Completed concat")
		D_z_c = tf.concat(axis=0, values=list_values[7])
		print("Completed concat")
		D_z_real = tf.concat(axis=0, values=list_values[8])
		print("Completed concat")
		D_z_hat_s = tf.concat(axis=0, values=list_values[9])
		print("Completed concat")
		D_z_s = tf.concat(axis=0, values=list_values[10])
		print("Completed concat")
		D_z_hat_t = tf.concat(axis=0, values=list_values[11])
		print("Completed concat")
		D_z_t = tf.concat(axis=0, values=list_values[12])
		
		losses = dict()
		with tf.variable_scope("losses"):
			losses["reconstruction"] = tf.sqrt(tf.reduce_mean(tf.square(x-x_hat)))
			losses["disc_image_classifier"] = self.gan_loss(D_z_c, True) + self.gan_loss(D_z_hat_c,False) + self.gan_loss(D_z_real, True)
			losses["gen_image_classifier"] = self.gan_loss(D_z_hat_c, True)
			losses["disc_text_classifier"] = self.gan_loss(D_z_t,True) + self.gan_loss(D_z_hat_t, False)
			losses["gen_text_classifier"] = self.gan_loss(D_z_hat_t, True)
			losses["disc_image_discriminator"] = self.gan_loss(D_x_gen,False) + self.gan_loss(D_x,True) + self.gan_loss(D_x_hat, False) + self.gan_loss(D_x_dash, False) + 2*self.gan_loss(D_x, True)
			losses["generator_image"] = self.gan_loss(D_x_gen, True) + (self.lambda_1*losses["reconstruction"])  + self.gan_loss(D_x_hat, True) + self.gan_loss(D_x_dash, True) 
			losses["generator_image_gan"] = self.gan_loss(D_x_gen, True) + (self.lambda_1*losses["reconstruction"])  + self.gan_loss(D_x_hat, True) + self.gan_loss(D_x_dash, True) 
			losses["text_encoder"] = losses["gen_text_classifier"] + (losses["reconstruction"]*self.lambda_1)
			losses["disc_style_classifier"] = self.gan_loss(D_z_hat_s,False) + self.gan_loss(D_z_s, True)
			losses["gen_style_classifier"] = self.gan_loss(D_z_hat_s, True)
			losses["encoder"] = losses["gen_image_classifier"] + (self.lambda_1*losses["reconstruction"]) + losses["gen_style_classifier"]
		print("Completed losses")
		variable_dict = dict()
		variable_dict["encoder"] = [i for i in filter(lambda x: x.name.startswith("encoder"), tf.trainable_variables())]
		variable_dict["text_encoder"] = [i for i in filter(lambda x: x.name.startswith("text_encoder"), tf.trainable_variables())]
		variable_dict["generator"] = [i for i in filter(lambda x: x.name.startswith("generator"), tf.trainable_variables())]
		variable_dict["image_disc"] = [i for i in filter(lambda x: x.name.startswith("image_disc"), tf.trainable_variables())]
		variable_dict["image_class"] = [i for i in filter(lambda x: x.name.startswith("image_class"), tf.trainable_variables())]
		variable_dict["text_class"] = [i for i in filter(lambda x: x.name.startswith("text_class"), tf.trainable_variables())]
		variable_dict["style_class"] = [i for i in filter(lambda x: x.name.startswith("style_class"), tf.trainable_variables())]
		print("Completed weights")
		optimizer = dict()
		with tf.variable_scope("optimizers"):
			# encoder_adam = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5,beta2=0.9)
			print("encoder")
			optimizer["encoder"] = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5,beta2=0.9).minimize(losses["encoder"], var_list=variable_dict["encoder"])
			print("text_encoder")
			optimizer["text_encoder"] = tf.train.AdamOptimizer(self.learning_rate[1], beta1=0.5, beta2=0.9).minimize(losses["text_encoder"], var_list=variable_dict["text_encoder"])
			print("generator")
			optimizer["generator"] = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5,beta2=0.9).minimize(losses["generator_image"], var_list=variable_dict["generator"])
			optimizer["generator_gan"] = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5,beta2=0.9).minimize(losses["generator_image_gan"], var_list=variable_dict["generator"])
			optimizer["generator_reconstruction"] = tf.train.AdamOptimizer(self.learning_rate[0],beta1=0.5,beta2=0.9).minimize(self.lambda_1*losses["reconstruction"], var_list=variable_dict["generator"])
			print("disc_image")
			optimizer["discriminator"] = tf.train.AdamOptimizer(self.learning_rate[3],beta1=0.5, beta2=0.9).minimize(losses["disc_image_discriminator"], var_list=variable_dict["image_disc"])
			print("disc_non_image")
			optimizer["code_discriminator"] = tf.train.AdamOptimizer(self.learning_rate[4],beta1=0.5, beta2=0.9).minimize(losses["disc_image_classifier"], var_list=variable_dict["image_class"])
			optimizer["text_discriminator"] = tf.train.AdamOptimizer(self.learning_rate[5],beta1=0.5, beta2=0.9).minimize(losses["disc_text_classifier"], var_list=variable_dict["text_class"])
			optimizer["style_discriminator"] = tf.train.AdamOptimizer(self.learning_rate[6],beta1=0.5, beta2=0.9).minimize(losses["disc_style_classifier"], var_list=variable_dict["style_class"])
		print("Completed optimizers")
		return placeholders, optimizer, losses, x_hat

epoch = 600
 # batch_size = 16
embedding_size =128
motion_size=4
num_class_image=13
frames=4
num_class_motion = 5

def save_visualization(X, nh_nw=(16,2+frames), save_path='../results/%s/sample.jpg'%(sys.argv[4])):
	X = morph(X)
	print(X.shape)
	h,w = X.shape[1], X.shape[2]
	img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))
	
	for n,x in enumerate(X):
		j = n // nh_nw[1]
		i = n % nh_nw[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = x[:,:,:3]
	np.save("%s.%s"%(save_path.split(".")[0],".npy"), img)
	scipy.misc.imsave(save_path, img)

def morph(X):
	batch_size = int(X.shape[0])
	dim_channel = int(X.shape[-1]) // (frames+2)
	h,w = map(lambda x: int(x), X.shape[1:3])
	img = np.zeros([(2+frames)*batch_size,h,w,dim_channel])
	for i in range(batch_size):
		for t in range(frames+2):
			img[i*(frames+2) + t] = X[i,:,:,t*dim_channel:t*dim_channel+dim_channel]
	return img

def random_label(batch_size):
	t = np.random.choice(10, batch_size, replace=True)
	random = np.zeros(shape=[batch_size,13])
	for i in range(batch_size):
		random[i, int(t[i])] = 1
		random[i, 10:] = np.random.randint(0,256,[3]).astype(float) / 255
	return random	

def random_motion_label(batch_size, num_class_motion):
    assert type(num_class_motion) == int
    t = np.random.choice(num_class_motion, batch_size, replace=True)
    random = np.zeros(shape=[batch_size, num_class_motion])
    for i in range(batch_size):
        random[i, int(t[i])] = 1
    return random

def train_epoch(flag=False, initial=True):
	diter = 5
	count =  0
	large_iter =  100
	if flag  :
		final_iter = large_iter
	else:
		final_iter = diter
	run=0
	start_time = time.time()
	loss_val = [0,0,0,0,0,0,0]
	while run <= num_examples:
		for t in range(final_iter):
			feed_list = gan.generate_batch()
			run += batch_size
			feed_dict = {
				placeholders['image_input'] : feed_list[0],
				placeholders['x'] : feed_list[1],
				placeholders['image_class_input'] : feed_list[2],
				placeholders['text_label_input'] : feed_list[3],
				placeholders['z_s'] : np.random.normal(0,1,[batch_size*frames, embedding_size]),
				placeholders['z_c'] : random_label(batch_size*frames),
#				placeholders['z_t'] : np.random.normal(0,1,[batch_size*frames, num_class_motion])
				placeholders['z_t'] : random_motion_label(batch_size*frames, num_class_motion)
			}
			if initial:
				_, loss_val[1] = session.run([optimizers["code_discriminator"], losses["disc_image_classifier"]], feed_dict=feed_dict)
				_, loss_val[2] = session.run([optimizers["text_discriminator"], losses["disc_text_classifier"]], feed_dict=feed_dict)
				_, loss_val[3] = session.run([optimizers["style_discriminator"], losses["disc_style_classifier"]], feed_dict=feed_dict)
			_, loss_val[0] = session.run([optimizers["discriminator"],losses["disc_image_discriminator"]], feed_dict=feed_dict)

		for _ in range(2*diter):
			feed_list = gan.generate_batch()
			run += batch_size
			feed_dict = {
				placeholders['image_input'] : feed_list[0],
				placeholders['x'] : feed_list[1],
				placeholders['image_class_input'] : feed_list[2],
				placeholders['text_label_input'] : feed_list[3],
				placeholders['z_s'] : np.random.normal(0,1,[batch_size*frames, embedding_size]),
				placeholders['z_c'] : random_label(batch_size*frames),
				placeholders['z_t'] : random_motion_label(batch_size*frames, num_class_motion)
#				placeholders['z_t'] : np.random.normal(0,1,[batch_size*frames, num_class_motion])
			}
			if initial :
				_, loss_val[6] = session.run([optimizers["generator"], losses["generator_image"]], feed_dict=feed_dict)
				_, loss_val[4] = session.run([optimizers["encoder"], losses["encoder"]], feed_dict=feed_dict)
				_, loss_val[5] = session.run([optimizers["text_encoder"], losses["text_encoder"]], feed_dict=feed_dict)
			else:
				_, loss_val[6] = session.run([optimizers["generator_gan"], losses["generator_image"]], feed_dict=feed_dict)

		# z_c = session.run(z_hat_c, feed_dict=feed_dict)
		count += 1
		if count % 10 == 0 or flag:
			print("%d:%d : "%(ep+1,run) + " : ".join(map(lambda x : str(x),loss_val)) + " " + str(time.time() - start_time))
			# print(z_c)
		start_time = time.time() 

image_sample,image_gen,image_labels, text_labels = generate(batch_size, frames)
save_visualization(np.concatenate([image_sample,image_gen],axis=3), save_path='../results/vae/64/frame_step_8/sample.jpg')
# save_visualization(image_gen, save_path='../results/vae/64/frame_step_8/sample_gen.jpg')
gan = VAEGAN(batch_size=batch_size, embedding_size=embedding_size, image_shape=[64,64,3], 
	num_class_motion=num_class_motion, num_class_image=num_class_image, frames=frames, video_create=True)

placeholders,optimizers, losses, x_hat = gan.build_model()
session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

saver = tf.train.Saver()
tf.global_variables_initializer().run()

print("Running code: ")

epoch = int(sys.argv[-1])
diter = 5
num_examples = 64000
for ep in range(epoch):
	if ep % 50 == 0 or ep < 7:
		if ep > 5:
			train_epoch(flag=True)
		else :
			train_epoch(flag=True, initial=True)
	else:
		train_epoch()
	print("Saving image")
	feed_list = gan.generate_batch()
	feed_dict = {
		placeholders['image_input'] : image_sample,
		placeholders['x'] : image_gen,
		placeholders['image_class_input'] : image_labels,
		placeholders['text_label_input'] : text_labels,
		placeholders['z_s'] : np.random.normal(0,1,[batch_size*frames, embedding_size]),
		placeholders['z_c'] : random_label(batch_size*frames),
		placeholders['z_t'] : random_motion_label(batch_size*frames, num_class_motion)
#		placeholders['z_t'] : np.random.normal(0,1,[batch_size*frames, num_class_motion])
	}
	images = session.run(x_hat, feed_dict=feed_dict)
	save_visualization(np.concatenate([image_sample, images],axis=3), save_path="../results/vae/64/frame_step_8/sample_%d.jpg"%(ep+1))
saver.save(session, "/media/hdd/hdd/frame_2_generator_vae.ckpt")
