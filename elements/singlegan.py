import tensorflow as tf
import numpy as np
import scipy.misc
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# LeakyRelu = tf.cosntrib.keras.layers.LeakyReLU()
def LeakyRelu(X,alpha=0.3):
	return alpha*X + (1-alpha)*tf.nn.relu(X)

class SingleGAN():
	def __init__ (self, batch_size = 50, image_shape = [28,28,1], embedding_size = 128, num_class =10, dim0=256,dim1 = 1024, dim2 = 128, dim3 = 64, dim_channel = 1, dim4=16, learning_rate_1=sys.argv[1], learning_rate_2=sys.argv[2], learning_rate_3=sys.argv[3], frames = 8):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.frames = frames
		self.dim0 = dim0
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim4 = dim4
		self.learning_rate_1 = float(learning_rate_1)
		self.learning_rate_2 = float(learning_rate_2)
		self.learning_rate_3 = float(learning_rate_3)
		self.dim_1 = self.image_shape[0]
		self.dim_2 = self.image_shape[0] // 2
		self.dim_4 = self.image_shape[0] // 4
		self.dim_8 = self.image_shape[0] // 8
		self.dim_channel = dim_channel
		self.device = "/gpu:0"
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=0.02)
	
	def learningR(self):
		return self.learning_rate_1 , self.learning_rate_2, self.learning_rate_3

	def normalize(self, X,reuse=False, name=None, flag=False):
		if not flag:
			mean, vari = tf.nn.moments(X, 0, keep_dims=True)
		else:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
		return tf.nn.batch_normalization(X, mean, vari, offset=None, scale=None, variance_epsilon=1e-6,name=name)

	def cross_entropy(self, X, flag=True):
		if flag:
			labels = tf.ones_like(X)
		else:
			labels = tf.zeros_like(X)
		softmax = tf.nn.softmax_cross_entropy_with_logits(logits=X, labels=labels)
		return tf.reduce_mean(softmax)

	def build_model(self):
		with tf.device("/gpu:0"):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size, self.frames, self.num_class])
			r_image = tf.placeholder(tf.float32,[self.batch_size, self.frames] + self.image_shape)
			fake_player, real_player, real_value, fake_value, _ = self.generate_video(embedding, classes, r_image)
			energy_lstm = tf.reduce_mean(tf.square(fake_player-real_player))
			real_value_softmax = tf.nn.softmax(real_value)
			energy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_value_softmax, logits=fake_value))
			d_cost = 0.7*(self.cross_entropy(real_value, True) + self.cross_entropy(fake_value, False)) - (0.3*energy)
			g_cost = 0.7*self.cross_entropy(fake_value, True) - (0.3*energy)
			return embedding, classes, r_image, d_cost, g_cost, fake_value, real_value, energy_lstm
	def generate_video(self, embedding, classes, r_image):
		with tf.variable_scope("lstm") as scope:
			self.lstm_setup(scope)
		real_value = []
		real_player = []
		fake_value = []
		fake_player = []
		embedding_current = embedding
		video = None
		flag = False
		for i in range(self.frames):
			embedding_current, fake_player_i, fake_value_i, real_player_i, real_value_i,g_image = self.frame_gen(embedding_current, classes[:,i], r_image[:,i],flag)
			flag = True
			fake_player.append(fake_player_i)
			fake_value.append(fake_value_i)
			real_player.append(real_player_i)
			real_value.append(real_value_i)
			if video == None:
				video = tf.reshape(g_image,shape=[self.batch_size, 1] + self.image_shape)
			else: 
				video = tf.concat([video, tf.reshape(g_image,shape=[self.batch_size, 1] + self.image_shape)], axis=1)
		fake_value = tf.reshape(tf.stack(fake_value),shape=[self.batch_size*self.frames, self.num_class])
		real_value = tf.reshape(tf.stack(real_value),shape=[self.batch_size*self.frames, self.num_class])
		fake_player = tf.reshape(tf.stack(fake_player),shape=[self.batch_size*self.frames, self.dim0])
		real_player = tf.reshape(tf.stack(real_player),shape=[self.batch_size*self.frames, self.dim0])
		return fake_player, real_player, real_value, fake_value, video

	def lstm_setup(self, scope):
		self.lstm = tf.contrib.rnn.BasicLSTMCell(self.dim0, reuse=scope.reuse)
		self.state = self.lstm.zero_state(batch_size, tf.float32)

	def lstm_layer(self, embedding,scope):
		cell_output, state_output = self.lstm(embedding, self.state)
		self.state = state_output
		return self.normalize(cell_output)

	def frame_gen(self, embedding, classes, r_image, flag=False):
		with tf.variable_scope("generator") as scope:
			if flag : 
				scope.reuse_variables()
			h4 = self.generate(embedding, classes, scope)
		g_image = h4
		with tf.variable_scope("discriminator") as scope:
			if flag :
				scope.reuse_variables()
			real_player, real_value = self.discriminate(r_image, classes, scope)
		with tf.variable_scope("discriminator") as scope:
			scope.reuse_variables()
			fake_player,fake_value = self.discriminate(g_image, classes, scope)
		# with tf.variable_scope("lstm") as scope:
			# embedding_return = self.lstm_layer(fake_player, scope)
		return embedding, fake_player, fake_value, real_player, real_value, g_image

	def discriminate(self, image, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size, 1,1, self.num_class])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_1, self.dim_1, self.num_class])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2, self.dim_2, self.num_class])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_4, self.dim_4, self.num_class])
		
			LeakyReLU = tf.contrib.keras.layers.LeakyReLU()

			# image_proc = self.normalize(tf.concat(axis=3,
				# values=[image, yneed_1]),flag=True)
			image_proc = tf.concat(axis=3,
				values=[self.normalize(image, flag=True),yneed_1])
			h1 = tf.layers.conv2d(image_proc, filters=self.dim4, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = LeakyReLU(h1)
			h1_concat = self.normalize(tf.concat(axis=3, values=[h1, yneed_2]))
			h2 = tf.layers.conv2d(h1_concat, filters=self.dim3, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None, 
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h2_relu = LeakyReLU(h2)
			h2_concat = self.normalize(tf.concat(axis=3, values=[h2_relu, yneed_3]))
			h3 = tf.layers.conv2d(h2_concat, filters=self.dim2, kernel_size=[5,5],
				strides=[1,1], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			h3_relu = LeakyReLU(h3)
			h3_reshape = tf.reshape(h3_relu, shape=[-1, self.dim_4*self.dim_4*self.dim2])
			h3_concat = self.normalize(tf.concat(axis=1, values=[h3_reshape, classes]),
				name="h3_concat_normalize", reuse=scope.reuse)
			h4 = tf.layers.dense(h3_concat, units=self.dim1, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_1',
				reuse=scope.reuse)
			h4_relu = LeakyReLU(h4)
			h4_concat = self.normalize(tf.concat(axis=1, values=[h4_relu, classes]),
				name="h4_concat_normalize",reuse=scope.reuse)
			h5 = tf.layers.dense(h4_concat, units=self.dim0, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			h5_relu = LeakyReLU(h5)
			h5_concat = self.normalize(tf.concat(axis=1, values=[h5_relu, classes]),
				name="h4_concat_normalize",reuse=scope.reuse)
			h6 = tf.layers.dense(h4_concat, units=num_class, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_3',
				reuse=scope.reuse)
			return self.normalize(h5_relu), LeakyReLU(self.normalize(h6,name="last_normalize",reuse=scope.reuse))

	def generate(self, embedding, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size,1, 1, self.num_class])
			embedding = tf.concat(axis=1, values=[embedding, classes])
			h1 = tf.layers.dense(embedding, units=self.dim1, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_concat = tf.concat(axis=1, values=[h1_relu, classes])
			h2 = tf.layers.dense(h1_concat, units=self.dim_8*self.dim_8*self.dim2, 
				activation=None, kernel_initializer=self.initializer,
				name='dense_2',	reuse=scope.reuse)
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = tf.concat(axis=3,
				values=[tf.reshape(h2_relu, shape=[self.batch_size,self.dim_8,self.dim_8,self.dim2]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_8, self.dim_8, 
				self.num_class])])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = self.dim3, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h3_relu = tf.nn.relu(self.normalize(h3,flag=True))
            # print(h3.get_shape())
			h3_concat = tf.concat(axis=3,
				values=[tf.reshape(h3_relu, shape=[self.batch_size,self.dim_4,self.dim_4,self.dim3]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_4, self.dim_4, self.num_class])])
			h4 = tf.layers.conv2d_transpose(inputs=h3_concat, filters = self.dim4, 
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
			return tf.nn.sigmoid(h5)

	def samples_generator(self):
		with tf.device("/gpu:0"):
			batch_size = self.batch_size
			embedding = tf.placeholder(tf.float32,[self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32,[self.batch_size, self.frames, self.num_class])
			list_images = []
			embedding_current = embedding
			# with tf.variable_scope("lstm") as scope:
			# 	scope.reuse_variables()
			# 	self.lstm_setup(scope)
			for i in range(self.frames):
				with tf.variable_scope("generator") as scope:
					scope.reuse_variables()
					t = self.generate(embedding_current,classes[:,i],scope)
					list_images.append(t)
				with tf.variable_scope("discriminator") as scope:
					scope.reuse_variables()
					fake_player, _ = self.discriminate(t, classes[:,i], scope)
				# with tf.variable_scope("lstm") as scope:
				# 	scope.reuse_variables()
				# 	embedding_current = self.lstm_layer(fake_player,scope)
			images = tf.reshape(tf.stack(list_images),shape=[self.batch_size*self.frames] + self.image_shape)
			return embedding,classes,images

# training part
epoch = 100
learning_rate = 1e-2
batch_size = 16
embedding_size = 256
num_class = 10
frames = 8

gan = SingleGAN(batch_size=batch_size, embedding_size=embedding_size, image_shape=[64,64,1], frames = frames, num_class=num_class)

embedding, vector, real_image, d_loss, g_loss, prob_fake, prob_real, lstm_loss = gan.build_model()
session  = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
# relevant weight list
g_weight_list = [i for i in (filter(lambda x: x.name.startswith("gen"),tf.trainable_variables()))]
d_weight_list = [i for i in (filter(lambda x: x.name.startswith("disc"),tf.trainable_variables()))]
lstm_weight_list = [i for i in (filter(lambda x: x.name.startswith("lstm"), tf.trainable_variables()))]
print(g_weight_list)
print(d_weight_list)
print(lstm_weight_list)
# optimizers
# with tf.device("/gpu:0"):
lr1, lr2, lr3 = gan.learningR()
g_optimizer = tf.train.AdamOptimizer(lr1,beta1=0.5).minimize(g_loss,var_list=g_weight_list)
d_optimizer = tf.train.AdamOptimizer(lr2,beta1=0.5).minimize(d_loss,var_list=d_weight_list)
# lstm_optimizer = tf.train.AdamOptimizer(lr3,  beta1=0.3).minimize(lstm_loss, var_list=lstm_weight_list)
saver = tf.train.Saver()

embedding_sample, vector_sample, image_sample = gan.samples_generator()

tf.global_variables_initializer().run()

def generate(batch_size, frames):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28, 28, 1])
	batch = np.zeros([batch_size,frames,64,64,1])
	batch_lab = np.zeros([batch_size, frames, 10])
	for i in range(frames):
		batch[:,i,2+(4*i):30+(4*i),2:30,:] = batch1
		batch_lab[:, i, :10] = batch1_labels
	return (batch, batch_lab)

def save_visualization(X, nh_nw, save_path='../results/singlegan/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

embedding_sample = np.random.uniform(-1,1,size=[batch_size,embedding_size])
vector_sample = np.zeros([batch_size,num_class])
rand = np.random.randint(0,num_class-1,batch_size)
for t in range(batch_size):
	vector_sample[t][rand[t]] = 1
sample_ = generate(batch_size, frames)
save_visualization(sample_[0].reshape([batch_size*frames, 64,64,1]), (8,16))
embedding_,vector_,image_sample = gan.samples_generator()

print('mnistsamples/sample_%d.jpg'%(batch_size))

for ep in range(epoch):
	start_time = time.time()
	average_loss_val = [0,0,0]
	lstm_loss_val = 0
	for t in range(16000 // batch_size):
		# print(t+1)
		batch = generate(batch_size,frames)
		random = np.random.uniform(-1,1,size=[batch_size,embedding_size]).astype(np.float32)
		feed_dict_1 = {
			real_image : batch[0],
			embedding : random,
			vector : batch[1]
		}
		feed_dict_2 = {
			real_image : batch[0],
			embedding : random,
			vector : batch[1]
		}
		# g_loss_val = 0
		if ep < 30:
			_,g_loss_val = session.run([g_optimizer,g_loss],feed_dict=feed_dict_2) 
			_,d_loss_val = session.run([d_optimizer,d_loss],feed_dict=feed_dict_1)
			average_loss_val[0] += g_loss_val
			average_loss_val[1] += d_loss_val
		# if ep > 3:
		# 	_,lstm_loss_val = session.run([lstm_optimizer,lstm_loss],feed_dict=feed_dict_2) 
		# 	average_loss_val[2] += lstm_loss_val
		if t%10 == 0 and t>0:
			print("Done with batches: " + str(t*batch_size) + "Losses :: Generator: " + str(average_loss_val[0]/10) + " and Discriminator: " + str(average_loss_val[0]/10) + " = " + str(average_loss_val[0]/10 + average_loss_val[1]/10))
			# if ep > 3:
			# 	print("LSTM loss : " + str(average_loss_val[2]/10))
			print(time.time()-start_time)
			start_time= time.time()
			average_loss_val = [0,0,0]
	print("Saving sample images and data for later testing for epoch: %d"%(ep+1))
	feed_dict = {
		real_image : batch[0],
		embedding_ : embedding_sample,
		vector_ : sample_[1]
	}
	gen_samples = session.run(image_sample,feed_dict=feed_dict)
	save_visualization(gen_samples,(16,8),save_path=('../results/singlegan/sample_%d.jpg'%(ep)))
	saver.save(session,'./dcgan.ckpt')
	print("Saved session")

print("Video GAN in under 400 lines! Done")
