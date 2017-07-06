import tensorflow as tf
import numpy as np
import scipy.misc
import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


def batch_normalize(X, eps=1e-6,flag=False):
	if flag : 
		if X.get_shape().ndims == 4:
			mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
			return tf.nn.batch_normalization(X,mean, vari, variance_epsilon=eps)
		elif X.get_shape().ndims == 2:
			mean, vari = tf.nn.moments(X, 0, keep_dims=True)
			return tf.nn.batch_normalization(X, mean, vari, variance_epsilon=eps)
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

# LeakyRelu = tf.cosntrib.keras.layers.LeakyReLU()
def LeakyRelu(X,alpha=0.3):
	return alpha*X + (1-alpha)*tf.nn.relu(X)

def lrelu(X):
	return LeakyRelu(X)

# def bce(o,t):
	# o  = tf.clip_by_value(o,1e-5,-1e-5)
	# return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o,labels=t))

class DCGAN():
	def __init__ (self, batch_size = 50, image_shape = [28,28,1], embedding_size = 128, num_class =10, dim1 = 1024, dim2 = 128, dim3 = 64, dim_channel = 1, dim4=16, learning_rate_1=sys.argv[1], learning_rate_2=sys.argv[2]):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim4 = dim4
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
		with tf.device("/gpu:0"):
			# self.g_weight1 = tf.Variable(tf.random_normal([embedding_size + num_class, dim1], stddev = 0.2), name="generator_weight1")
			# self.g_weight2 = tf.Variable(tf.random_normal([dim1 + num_class, dim2*7*7], stddev = 0.2), name="generator_weight2")
			# self.g_weight3 = tf.Variable(tf.random_normal([5,5,dim3,dim2+num_class], stddev = 0.2), name="generator_weight3")
			# self.g_weight4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim3+num_class], stddev = 0.2), name="generator_weight4")

			self.d_weight1 = tf.Variable(tf.random_normal([5,5,dim_channel+num_class, dim3],stddev = 0.2), name="disc_weight1")
			self.d_weight2 = tf.Variable(tf.random_normal([5,5,dim3+num_class, dim2],stddev = 0.2), name="disc_weight2")
			self.d_weight3 = tf.Variable(tf.random_normal([dim2*7*7+num_class, dim1],stddev = 0.2), name="disc_weight3")
			self.d_weight4 = tf.Variable(tf.random_normal([dim1+num_class,1],stddev = 0.2), name="disc_weight4")
	
	def learningR(self):
		return self.learning_rate_1 , self.learning_rate_2

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
			classes = tf.placeholder(tf.float32, [self.batch_size,self.num_class])
			r_image = tf.placeholder(tf.float32,[self.batch_size] + self.image_shape)
			real_image = tf.reshape(r_image,[self.batch_size] + self.image_shape)
			with tf.variable_scope("generator") as scope:	
				h4 = self.generate(embedding,classes,scope)
			g_image = h4
			with tf.variable_scope("discriminator") as scope:
				real_value = self.discriminate(real_image,classes,scope)
			# prob_real = tf.nn.sigmoid(real_value)
			with tf.variable_scope("discriminator") as scope:
				scope.reuse_variables()
				fake_value = self.discriminate(g_image,classes,scope)
			# prob_fake = tf.nn.sigmoid(fake_value)
			d_cost = self.cross_entropy(real_value, True) + self.cross_entropy(fake_value, False)
			g_cost = self.cross_entropy(fake_value, True)
			# d_cost = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
			# g_cost = -tf.reduce_mean(tf.log(prob_fake))
			return embedding, classes, r_image, d_cost, g_cost, fake_value, real_value

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
			h5 = tf.layers.dense(h4_concat, units=num_class, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return LeakyReLU(self.normalize(h5,name="last_normalize",reuse=scope.reuse))

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
			embedding = tf.placeholder(tf.float32,[batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32,[batch_size,self.num_class])
			with tf.variable_scope("generator") as scope:
				scope.reuse_variables()
				t = self.generate(embedding,classes,scope)
			return embedding,classes,t

# training part
epoch = 100
learning_rate = 1e-2
batch_size = 64
embedding_size = 256
num_class = 10

gan = DCGAN(batch_size=batch_size, embedding_size=embedding_size, image_shape=[64,64,1])

embedding, vector, real_image, d_loss, g_loss, prob_fake, prob_real = gan.build_model()
session  = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
# relevant weight list
g_weight_list = [i for i in (filter(lambda x: x.name.startswith("gen"),tf.trainable_variables()))]
d_weight_list = [i for i in (filter(lambda x: x.name.startswith("disc"),tf.trainable_variables()))]
print(g_weight_list)
print(d_weight_list)
# optimizers
# with tf.device("/gpu:0"):
lr1, lr2 = gan.learningR()
g_optimizer = tf.train.AdamOptimizer(lr1,beta1=0.5).minimize(g_loss,var_list=g_weight_list)
d_optimizer = tf.train.AdamOptimizer(lr2,beta1=0.5).minimize(d_loss,var_list=d_weight_list)
saver = tf.train.Saver()

embedding_sample, vector_sample, image_sample = gan.samples_generator()

tf.global_variables_initializer().run()

def generate(batch_size):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28, 28, 1])
	batch2, batch2_labels = mnist.train.next_batch(batch_size)
	batch2 = batch2.reshape([batch_size, 28, 28, 1])
	batch = np.zeros([batch_size,64,64,1])
	batch[:,2:30,2:30,:] = batch1
	batch[:,34:62,34:62,:] = batch2
	return (batch, batch1_labels + batch2_labels)

def save_visualization(X, nh_nw, save_path='../results/dcgan64/sample.jpg'):
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
sample_ = generate(batch_size)
save_visualization(sample_[0], (8,8))
embedding_,vector_,image_sample = gan.samples_generator()

print('mnistsamples/sample_%d.jpg'%(batch_size))

for ep in range(epoch):
	for t in range(64000 // batch_size):
		# print(t+1)
		batch = generate(batch_size)
		random = np.random.uniform(-1,1,size=[batch_size,embedding_size]).astype(np.float32)
		feed_dict_1 = {
			real_image : batch[0],
			embedding : random,
			vector : batch[1]
		}
		feed_dict_2 = {
			# real_image : batch[0],
			embedding : random,
			vector : batch[1]
		}
		# g_loss_val = 0
		_,g_loss_val = session.run([g_optimizer,g_loss],feed_dict=feed_dict_2) 
		_,d_loss_val = session.run([d_optimizer,d_loss],feed_dict=feed_dict_1)
		if t%10 == 0 and t>0:
			print("Done with batches: " + str(t*batch_size) + "Losses :: Generator: " + str(g_loss_val) + " and Discriminator: " + str(d_loss_val) + " = " + str(d_loss_val + g_loss_val))
	print("Saving sample images and data for later testing")
	feed_dict = {
		# real_image : batch[0],
		embedding_ : embedding_sample,
		vector_ : vector_sample
	}
	gen_samples = session.run(image_sample,feed_dict=feed_dict)
	save_visualization(gen_samples,(8,8),save_path=('../results/dcgan64/sample_%d.jpg'%(ep)))
	saver.save(session,'./dcgan.ckpt')
	print("Saved session")

