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

class DCGAN():
	def __init__ (self, batch_size = 50, image_shape = [28,28,1], embedding_size = 128, frames=8, num_class =32, num_class_input=12, dim1 = 1024, dim2 = 128, dim3 = 64, dim_channel = 1, dim4=16, learning_rate_1=sys.argv[1], learning_rate_2=sys.argv[2]):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.num_class_input = num_class_input
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim4 = dim4
		self.frames = frames
		self.learning_rate_1 = float(learning_rate_1)
		self.learning_rate_2 = float(learning_rate_2)
		self.dim_1 = self.image_shape[0]
		self.dim_2 = self.image_shape[0] // 2
		self.dim_4 = self.image_shape[0] // 4
		self.dim_8 = self.image_shape[0] // 8
		self.dim_channel = dim_channel
		self.device = "/gpu:0"
		self.image_size = reduce(lambda x,y : x*y, image_shape)
		self.initializer = tf.random_normal_initializer(stddev=0.02)
	
	def learningR(self):
		return self.learning_rate_1 , self.learning_rate_2

	def normalize(self, X,reuse=False, name=None, flag=False,local=False):
		if not flag:
			mean, vari = tf.nn.moments(X, 0, keep_dims=True)
		elif local :
			mean, vari = tf.nn.moments(X,[0,1], keep_dims=True)
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
			embedding = tf.placeholder(tf.float32, [self.batch_size,self.frames, self.embedding_size])
			class_input = tf.placeholder(tf.float32, [self.batch_size, self.frames,self.num_class_input])
			classes = self.bilstm(class_input)
			r_image = tf.placeholder(tf.float32,[self.batch_size,self.frames] + self.image_shape)
			real_image = tf.reshape(r_image,[self.batch_size*self.frames] + self.image_shape)
			embedding_reshape = tf.reshape(embedding,shape=[self.batch_size*self.frames, self.embedding_size])
			with tf.variable_scope("generator") as scope:	
				h4 = self.generate(embedding_reshape,classes,scope)
			g_image = h4
			with tf.variable_scope("discriminator") as scope:
				real_value = self.discriminate(real_image,classes,scope)
			# prob_real = tf.nn.sigmoid(real_value)
			with tf.variable_scope("discriminator") as scope:
				scope.reuse_variables()
				fake_value = self.discriminate(g_image,classes,scope)
			# prob_fake = tf.nn.sigmoid(fake_value)
		#	real_value_softmax = tf.nn.softmax(real_value)
		#	energy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_value_softmax, logits=fake_value))
			d_cost = self.cross_entropy(real_value, True) + self.cross_entropy(fake_value, False)
			g_cost = self.cross_entropy(fake_value, True)
			# d_cost = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
			# g_cost = -tf.reduce_mean(tf.log(prob_fake))
			return embedding, class_input, r_image, d_cost, g_cost, fake_value, real_value

	def bilstm(self, class_input,flag=False):
		assert num_class % 2 == 0
		with tf.variable_scope("lstm_front") as scope:
			if flag :
				scope.reuse_variables()
			self.lstmf = tf.contrib.rnn.BasicLSTMCell(self.num_class // 2, reuse=scope.reuse)
			self.statef = self.lstmf.zero_state(batch_size, tf.float32)
		with tf.variable_scope("lstm_back") as scope:
			if flag :
				scope.reuse_variables()
			self.lstmb = tf.contrib.rnn.BasicLSTMCell(self.num_class // 2, reuse=scope.reuse)
			self.stateb = self.lstmb.zero_state(batch_size, tf.float32)
		for i in range(self.frames):
			with tf.variable_scope("lstm_front") as scope:
				if i > 0 or flag:
					scope.reuse_variables()
				cell_output_f, self.statef = self.lstmf(class_input[:,i], self.statef)
			with tf.variable_scope("lstm_back") as scope:
				if i > 0 or flag:
					scope.reuse_variables()
				cell_output_b, self.statef = self.lstmb(class_input[:,self.frames - i - 1], self.stateb)
			required_front = tf.reshape(cell_output_f, shape=[self.batch_size, 1, self.num_class // 2])
			required_back = tf.reshape(cell_output_b, shape=[self.batch_size, 1, self.num_class // 2])
			if i == 0:
				return_class_f = required_front
				return_class_b = required_back
			else:
				return_class_f = tf.concat(values=[return_class_f, required_front],axis=1)
				return_class_b = tf.concat(values=[required_back, return_class_b],axis=1)
		class_embedding = tf.reshape(tf.concat(values=[return_class_f, return_class_b],axis=2),shape=[self.batch_size*self.frames, self.num_class])
		assert map(lambda x: int(x), class_embedding.get_shape()) == [self.batch_size*self.frames, self.num_class]
		return self.normalize(class_embedding,local=True)

	def discriminate(self, image, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size*self.frames, 1,1, self.num_class])
			yneed_1 = ystack*tf.ones([self.batch_size*self.frames, self.dim_1, self.dim_1, self.num_class])
			yneed_2 = ystack*tf.ones([self.batch_size*self.frames, self.dim_2, self.dim_2, self.num_class])
			yneed_3 = ystack*tf.ones([self.batch_size*self.frames, self.dim_4, self.dim_4, self.num_class])
		
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
			h5 = tf.layers.dense(h4_concat, units=num_class, 
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
			return LeakyReLU(self.normalize(h6,name="last_normalize",reuse=scope.reuse))

	def generate(self, embedding, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size*self.frames,1, 1, self.num_class])
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
				values=[tf.reshape(h2_relu, shape=[self.batch_size*self.frames,self.dim_8,self.dim_8,self.dim2]), 
				ystack*tf.ones(shape=[self.batch_size*self.frames, self.dim_8, self.dim_8, 
				self.num_class])])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = self.dim3, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h3_relu = tf.nn.relu(self.normalize(h3,flag=True))
            # print(h3.get_shape())
			h3_concat = tf.concat(axis=3,
				values=[tf.reshape(h3_relu, shape=[self.batch_size*self.frames,self.dim_4,self.dim_4,self.dim3]), 
				ystack*tf.ones(shape=[self.batch_size*self.frames, self.dim_4, self.dim_4, self.num_class])])
			h4 = tf.layers.conv2d_transpose(inputs=h3_concat, filters = self.dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h4_relu = tf.nn.relu(self.normalize(h4,flag=True))
			h4_concat = tf.concat(axis=3,
				values=[tf.reshape(h4_relu, shape=[self.batch_size*self.frames,self.dim_2,self.dim_2,self.dim4]), 
				ystack*tf.ones(shape=[self.batch_size*self.frames, self.dim_2, self.dim_2, self.num_class])])
			h5 = tf.layers.conv2d_transpose(inputs=h4_concat, filters = 5*self.dim_channel, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			h5_relu = tf.nn.relu(self.normalize(h5, flag=True))
			h5_concat = tf.concat(axis=3, 
				values=[h5_relu, ystack*tf.ones(shape=[self.batch_size*self.frames, self.dim_1, self.dim_1, self.num_class])])
			h6 = tf.layers.conv2d_transpose(inputs=h5_concat, filters = self.dim_channel,
				kernel_size=[5,5], strides=[1,1], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_4")
			return tf.nn.sigmoid(h6)

	def samples_generator(self):
		with tf.device("/gpu:0"):
			batch_size = self.batch_size
			embedding = tf.placeholder(tf.float32,[self.batch_size, self.frames, self.embedding_size])
			class_input = tf.placeholder(tf.float32,[self.batch_size,self.frames,self.num_class_input])
			classes = self.bilstm(class_input, flag=True)
			embedding_reshape = tf.reshape(embedding, shape=[self.batch_size*self.frames, self.embedding_size])
			with tf.variable_scope("generator") as scope:
				scope.reuse_variables()
				t = self.generate(embedding_reshape,classes,scope)
			return embedding,class_input,t

# training part
epoch = 100
learning_rate = 1e-2
batch_size = 16
frames = 4
embedding_size = 256
num_class = 32
num_class_input = 12

gan = DCGAN(batch_size=batch_size, embedding_size=embedding_size, image_shape=[32,32,1], num_class=num_class, num_class_input=num_class_input, frames=4)

embedding, vector, real_image, d_loss, g_loss, prob_fake, prob_real = gan.build_model()
session  = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
# relevant weight list
g_weight_list = [i for i in (filter(lambda x: x.name.startswith("gen"),tf.trainable_variables()))]
d_weight_list = [i for i in (filter(lambda x: x.name.startswith("disc"),tf.trainable_variables()))]
lstm_weight_list = [i for i in (filter(lambda x: x.name.startswith("lstm"), tf.trainable_variables()))]
print(g_weight_list)
print(d_weight_list)
# optimizers
# with tf.device("/gpu:0"):
lr1, lr2 = gan.learningR()
g_optimizer = tf.train.AdamOptimizer(lr1,beta1=0.5).minimize(g_loss,var_list=g_weight_list)
d_optimizer = tf.train.AdamOptimizer(lr2,beta1=0.5).minimize(d_loss,var_list=d_weight_list)
lstm_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(g_loss, var_list=g_weight_list+lstm_weight_list)
saver = tf.train.Saver()

embedding_sample, vector_sample, image_sample = gan.samples_generator()


def generate(batch_size):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28, 28])
	batch = np.zeros([batch_size,frames,32,32,1])
	batch_labels = np.zeros([batch_size, frames, num_class_input])
	random = np.random.randint(0,2,[batch_size,frames]).reshape(batch_size, frames,1)
	batch_labels[:,:,10:12] = np.concatenate([random,1-random],axis=2)
	for i in range(frames):
		batch_labels[:,i,:10] = batch1_labels
		batch[:,i,2:30,2:30,0] = batch1
		for t in range(batch_size):
			if batch_labels[t,i,10] == 1:
				# print("here")
				batch1[t] = np.rot90(batch1[t])
			else:
				batch1[t] = np.rot90(batch1[t],k=3)
	return (batch, batch_labels)

def save_visualization(X, nh_nw=[batch_size, frames], save_path='../results/sample_lstm/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    np.save("%s.%s"%(save_path.split(".")[0],".npy"), img)
    scipy.misc.imsave(save_path, img)

embedding_sample = np.random.uniform(-1,1,size=[batch_size,frames,embedding_size])
vector_sample = np.zeros([batch_size,num_class])
rand = np.random.randint(0,num_class-1,batch_size)
for t in range(batch_size):
	vector_sample[t][rand[t]] = 1
sample_ = generate(batch_size)
save_visualization(sample_[0].reshape([batch_size*frames,32,32,1]), (8,8))
vector_sample = sample_[1]
embedding_,vector_,image_sample = gan.samples_generator()

print('mnistsamples/sample_%d.jpg'%(batch_size))
tf.global_variables_initializer().run()

for ep in range(epoch):
	average_loss = [0,0]
	start_time = time.time()
	start_cycle = time.time()
	for t in range(64000 // batch_size):
		# print(t+1)
		batch = generate(batch_size)
		random = np.random.uniform(-1,1,size=[batch_size,frames,embedding_size]).astype(np.float32)
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
		if ep < 200:
			_,g_loss_val = session.run([lstm_optimizer,g_loss],feed_dict=feed_dict_2) 
		else :
			_, g_loss_val = session.run([g_optimizer,g_loss],feed_dict_2)
		_,d_loss_val = session.run([d_optimizer,d_loss],feed_dict=feed_dict_1)
		
		if t%10 == 0 and t>0:
			print(str(t*batch_size) + " :: G: " + str(g_loss_val) + " +  D: " + str(d_loss_val) + " = " + str(d_loss_val + g_loss_val) + " in " + str(time.time()-start_cycle))
			start_cycle = time.time()
	print("Saving sample images and data for later testing for epoch: %d"%(ep+1))
	feed_dict = {
		embedding_ : embedding_sample,
		vector_ : vector_sample
	}
	gen_samples = session.run(image_sample,feed_dict=feed_dict)
	save_visualization(gen_samples,(8,8),save_path=('../results/sample_lstm/sample_%d.jpg'%(ep)))
	saver.save(session,'./dcgan.ckpt')
	print(time.time() - start_time)
	print("Saved session")

