import tensorflow as tf
import numpy as np
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/hdd/hdd/data_backup/MNIST/",one_hot=True)


def batch_normalize(X, eps=1e-6):
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

def lrelu(X, leak = 0.2):
	f1 = (1 + leak)*0.5
	f2 = (1 - leak)*0.5
	return X*f1 + abs(X)*f2

def bce(o,t):
	o  = tf.clip_by_value(o,1e-5,-1e-5)
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o,labels=t))

class DCGAN():
	def __init__ (self, batch_size = 50, image_shape = [28,28,1], embedding_size = 128, num_class =10, dim1 = 1024, dim2 = 128, dim3 = 64, dim_channel = 1):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.num_class = num_class
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim_channel = dim_channel
		with tf.device("/gpu:0"):
			self.g_weight1 = tf.Variable(tf.random_normal([embedding_size + num_class, dim1], stddev = 0.2), name="generator_weight1")
			self.g_weight2 = tf.Variable(tf.random_normal([dim1 + num_class, dim2*7*7], stddev = 0.2), name="generator_weight2")
			self.g_weight3 = tf.Variable(tf.random_normal([5,5,dim3,dim2+num_class], stddev = 0.2), name="generator_weight3")
			self.g_weight4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim3+num_class], stddev = 0.2), name="generator_weight4")

			self.d_weight1 = tf.Variable(tf.random_normal([5,5,dim_channel+num_class, dim3],stddev = 0.2), name="disc_weight1")
			self.d_weight2 = tf.Variable(tf.random_normal([5,5,dim3+num_class, dim2],stddev = 0.2), name="disc_weight2")
			self.d_weight3 = tf.Variable(tf.random_normal([dim2*7*7+num_class, dim1],stddev = 0.2), name="disc_weight3")
			self.d_weight4 = tf.Variable(tf.random_normal([dim1+num_class,1],stddev = 0.2), name="disc_weight4")

	def build_model(self):
		with tf.device("/gpu:0"):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size,self.num_class])
			r_image = tf.placeholder(tf.float32,[self.batch_size,784])
			real_image = tf.reshape(r_image,[self.batch_size] + self.image_shape)
			h4 = self.generate(embedding,classes)
			g_image = tf.nn.sigmoid(h4)
			real_value = self.discriminate(real_image,classes)
			prob_real = tf.nn.sigmoid(real_value)
			fake_value = self.discriminate(g_image,classes)
			prob_fake = tf.nn.sigmoid(fake_value)
			# d_cost = bce(real_value, tf.ones_like(real_value)) + bce(fake_value,tf.zeros_like(fake_value))
			# g_cost = bce(fake_value, tf.ones_like(fake_value))
			d_cost = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
			g_cost = -tf.reduce_mean(tf.log(prob_fake))
			return embedding, classes, r_image, d_cost, g_cost, prob_fake, prob_real

	def discriminate(self, image, classes):
		with tf.device("/gpu:0"):
			ystack = tf.reshape(classes, tf.stack([self.batch_size, 1,1, self.num_class]))
			yneed = ystack*tf.ones([self.batch_size,28,28,self.num_class])
			yneed2 = ystack*tf.ones([self.batch_size,14,14,self.num_class])
			# print(yneed.shape)
			image_ = batch_normalize(image)
			# print(image_.shape)
			proc_image = tf.concat(axis=3, values=[image_, yneed])
			# print(proc_image.shape)
			h1 = lrelu(tf.nn.conv2d(proc_image, self.d_weight1, strides=[1,2,2,1],padding='SAME'))
			h1 = batch_normalize(tf.concat(axis=3, values=[h1, yneed2]))
			h2 = lrelu(tf.nn.conv2d(h1, self.d_weight2, strides=[1,2,2,1],padding='SAME'))
			h3 = tf.reshape(h2,[self.batch_size,-1])
			h4 = tf.concat(axis=1,values=[h3,classes])
			h5 = lrelu(batch_normalize(tf.matmul(h4, self.d_weight3)))
			h6 = tf.concat(axis=1,values=[h5,classes])
			h7 = lrelu(batch_normalize(tf.matmul(h6, self.d_weight4)))
			return h7

	def generate(self, embedding, classes):
		with tf.device("/gpu:0"):
			ystack = tf.reshape(classes, [self.batch_size, 1,1, self.num_class])
			embedding = tf.concat(axis=1,values=[embedding,classes])
			h1 = tf.nn.relu(batch_normalize(tf.matmul(embedding,self.g_weight1)))
			h1 = tf.concat(axis=1,values=[h1,classes])
			h2 = tf.nn.relu(batch_normalize(tf.matmul(h1,self.g_weight2)))
			h2 = tf.reshape(h2, [self.batch_size,7,7,self.dim2])
			h2 = tf.concat(axis=3,values=[h2,ystack*tf.ones([self.batch_size,7,7,self.num_class])])

			output_shape1 = [self.batch_size,14,14,self.dim3]
			h3 = tf.nn.conv2d_transpose(h2,self.g_weight3,output_shape=output_shape1,strides=[1,2,2,1])
			h3 = tf.nn.relu(batch_normalize(h3))
			h3 = tf.concat(axis=3,values=[h3,ystack*tf.ones([self.batch_size,14,14,self.num_class])])

			output_shape2 = [self.batch_size,28,28,self.dim_channel]
			h4 = tf.nn.conv2d_transpose(h3,self.g_weight4,output_shape=output_shape2,strides=[1,2,2,1])
			return batch_normalize(h4)

	def samples_generator(self):
		with tf.device("/gpu:0"):
			batch_size = self.batch_size
			embedding = tf.placeholder(tf.float32,[batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32,[batch_size,self.num_class])

			t = tf.nn.sigmoid(self.generate(embedding,classes))
			return embedding,classes,t

# training part
epoch = 1000
learning_rate = 8e-3

gan = DCGAN()

embedding, vector, real_image, d_loss, g_loss, prob_fake, prob_real = gan.build_model()
session  = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
# relevant weight list
g_weight_list = [i for i in (filter(lambda x: x.name.startswith("gen"),tf.trainable_variables()))]
d_weight_list = [i for i in (filter(lambda x: x.name.startswith("disc"),tf.trainable_variables()))]
print(g_weight_list)
print(d_weight_list)
# optimizers
# with tf.device("/gpu:0"):
g_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(g_loss,var_list=g_weight_list)
d_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(d_loss,var_list=d_weight_list)
saver = tf.train.Saver()

embedding_sample, vector_sample, image_sample = gan.samples_generator()

tf.global_variables_initializer().run()

def save_visualization(X, nh_nw, save_path='./mnistimages/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

batch_size = 50
embedding_size = 128
num_class = 10

embedding_sample = np.random.uniform(-1,1,size=[batch_size,embedding_size])
vector_sample = np.zeros([batch_size,num_class])
rand = np.random.randint(0,num_class-1,batch_size)
for t in range(batch_size):
	vector_sample[t][rand[t]] = 1

embedding_,vector_,image_sample = gan.samples_generator()

print('mnistsamples/sample_%d.jpg'%(batch_size))

for ep in range(epoch):
	for t in range(mnist.train.num_examples // batch_size):
		# print(t+1)
		batch = mnist.train.next_batch(batch_size)
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
	save_visualization(gen_samples,(14,14),save_path=('../results/dcgan_old/sample_%d.jpg'%(ep)))
	saver.save(session,'./dcgan.ckpt')
	print("Saved session")

