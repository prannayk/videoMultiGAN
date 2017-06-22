import tensorflow as tf
import numpy as np
# import scipy

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

def save_visualization(X, nh_nw, save_path='./mnistvideo/sample.jpg',flag=False):
	h,w = X.shape[1], X.shape[2]
	img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

	for n,x in enumerate(X):
	    j = n // nh_nw[1]
	    i = n % nh_nw[1]
	    img[j*h:j*h+h, i*w:i*w+w, :] = x
	if not flag:
		img = np.mean(img,axis=2)
		# scipy.misc.imsave(save_path,img)   	
	if flag:
		img1,img2 = np.split(img,axis=2, indices_or_sections=2)
		save_path = save_path.split(".")[0] + "_1" + save_path.split(".")[1]
		# scipy.misc.imsave(save_path, img1) 
		save_path = save_path.split(".")[0] + "_1" + save_path.split(".")[1]
		# scipy.misc.imsave(save_path, img2) 


class DCGAN():
	def __init__ (self, batch_size = 50, image_shape = [28,28,1], embedding_size = 128, text_embedding_size =10, dim1 = 1024, dim2 = 128, dim3 = 64, dim_channel = 1, name = "gan1",splices=7):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.text_embedding_size = text_embedding_size
		self.dim1 = dim1
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim_channel = dim_channel
		self.name = name
		self.splices = splices
		## assumption is square images
		self.dim_4 = image_shape[0] // 4
		self.dim_2 = image_shape[0] // 2
		self.image_input_size = image_shape[0]*image_shape[1]*image_shape[2]
		self.g_weight1 = tf.Variable(tf.random_normal([embedding_size + text_embedding_size, dim1], stddev = 0.2), name=(self.name+"_generator_weight1"))
		self.g_weight2 = tf.Variable(tf.random_normal([dim1 + text_embedding_size, dim2*self.dim_4*self.dim_4], stddev = 0.2), name=(self.name+"_generator_weight2"))
		self.g_weight3 = tf.Variable(tf.random_normal([5,5,dim3,dim2+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weight3"))
		self.g_weight4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim3+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weight4"))

		self.d_weight1 = tf.Variable(tf.random_normal([5,5,dim_channel+text_embedding_size, dim3],stddev = 0.2), name=(self.name+"_disc_weight1"))
		self.d_weight2 = tf.Variable(tf.random_normal([5,5,dim3+text_embedding_size, dim2],stddev = 0.2), name=(self.name+"_disc_weight2"))
		self.d_weight3 = tf.Variable(tf.random_normal([dim2*self.dim_4*self.dim_4+text_embedding_size, dim1],stddev = 0.2), name=(self.name+"_disc_weight3"))
		self.d_weight4 = tf.Variable(tf.random_normal([dim1+text_embedding_size,1],stddev = 0.2), name=(self.name+"_disc_weight4"))

	def build_model(self):
		prob_fake_list = list()
		prob_real_list = list()
		prob_real_wrong_list = list()
		# placeholder list
		embedding_list = list()
		classes_list = list()
		classes_fake_list = list()
		r_image_list = list()
		for t in range(2*self.splices):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size,self.text_embedding_size])
			classes_fake = tf.placeholder(tf.float32, [self.batch_size,self.text_embedding_size])
			classes_list.append(classes)
			embedding_list.append(embedding)
			classes_fake_list.append(classes_fake)
			r_image = tf.placeholder(tf.float32,[self.batch_size,self.image_input_size])
			r_image_list.append(r_image)
			real_image = tf.reshape(r_image,[self.batch_size] + self.image_shape)
			h4 = self.generate(embedding,classes)
			g_image = tf.nn.sigmoid(h4)
			real_value = self.discriminate(real_image,classes)
			real_wrong_value = self.discriminate(real_image,classes_fake)
			prob_real = tf.nn.sigmoid(real_value)
			prob_real_wrong = tf.nn.sigmoid(real_wrong_value)
			fake_value = self.discriminate(g_image,classes)
			prob_fake = tf.nn.sigmoid(fake_value)
			prob_fake_list.append(prob_fake)
			prob_real_wrong_list.append(prob_real_wrong)
			prob_real_list.append(prob_real)
		d_prob_result_fake = prob_fake_list[0]
		d_prob_result_real = prob_real_list[0]
		d_prob_result_fake_text = prob_real_wrong_list[0]
		g_fake = prob_fake_list[0]
		total_prob = prob_real_list[0] + prob_fake_list[0] + prob_real_wrong_list[0]
		for i in range(1,2*self.splices):
			d_prob_result_fake = prob_fake_list[i]*d_prob_result_fake
			d_prob_result_real = prob_real_list[i]*d_prob_result_real
			d_prob_result_fake_text = prob_real_wrong_list[i]*d_prob_result_fake_text
			g_fake = prob_fake_list[i] + g_fake
			total_prob += prob_real_list[i] + prob_fake_list[i] + prob_real_wrong_list[i]
		d_loss = tf.reduce_mean(tf.log(d_prob_result_real) + tf.log(1 - d_prob_result_fake_text) + tf.log(1 - d_prob_result_fake))
		g_loss = tf.reduce_mean(tf.log(tf.div(g_fake,total_prob)) + tf.log(1 - d_prob_result_real))

		return embedding_list, classes_list, classes_fake_list, r_image_list, g_loss, d_loss 

	def discriminate(self, image, classes):
		ystack = tf.reshape(classes, tf.stack([self.batch_size, 1,1, self.text_embedding_size]))
		yneed = ystack*tf.ones([self.batch_size,self.image_shape[0],self.image_shape[1],self.text_embedding_size])
		yneed2 = ystack*tf.ones([self.batch_size,self.dim_2,self.dim_2,self.text_embedding_size])
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
		ystack = tf.reshape(classes, [self.batch_size, 1,1, self.text_embedding_size])
		embedding = tf.concat(axis=1,values=[embedding,classes])
		h1 = tf.nn.relu(batch_normalize(tf.matmul(embedding,self.g_weight1)))
		h1 = tf.concat(axis=1,values=[h1,classes])
		h2 = tf.nn.relu(batch_normalize(tf.matmul(h1,self.g_weight2)))
		h2 = tf.reshape(h2, [self.batch_size,self.dim_4,self.dim_4,self.dim2])
		h2 = tf.concat(axis=3,values=[h2,ystack*tf.ones([self.batch_size,self.dim_4,self.dim_4,self.text_embedding_size])])

		output_shape1 = [self.batch_size,self.dim_2,self.dim_2,self.dim3]
		h3 = tf.nn.conv2d_transpose(h2,self.g_weight3,output_shape=output_shape1,strides=[1,2,2,1])
		h3 = tf.nn.relu(batch_normalize(h3))
		h3 = tf.concat(axis=3,values=[h3,ystack*tf.ones([self.batch_size,self.dim_2,self.dim_2,self.text_embedding_size])])

		output_shape2 = [self.batch_size,self.image_shape[0],self.image_shape[0],self.dim_channel]
		h4 = tf.nn.conv2d_transpose(h3,self.g_weight4,output_shape=output_shape2,strides=[1,2,2,1])
		return batch_normalize(h4)

	def samples_generator(self):
		batch_size = self.batch_size
		embedding = tf.placeholder(tf.float32,[batch_size, self.embedding_size])
		classes = tf.placeholder(tf.float32,[batch_size,self.text_embedding_size])

		t = tf.nn.sigmoid(self.generate(embedding,classes))
		return embedding,classes,t

class multiGAN():
	def __init__(self,splices=6,video_shape=[32,32,3],tree_style=1, batch_size=None, embedding_size=None, text_embedding_size=None,name=None):
		if tree_style != 1:
			raise NotImplemented
		self.tree_style = tree_style
		self.video_shape = video_shape
		self.splices = splices
		if batch_size  != None:
			self.batch_size = batch_size
		if embedding_size  != None:
			self.embedding_size = embedding_size
		if text_embedding_size  != None:
			self.text_embedding_size = text_embedding_size
		if name  != None:
			self.name = name

	def build_tree(self):
		if self.tree_style == 1:
			video_shape = list(self.video_shape)
			video_shape[2]*=2
			self.crossover_gan = DCGAN(batch_size=self.batch_size,image_shape=video_shape, embedding_size=self.embedding_size, text_embedding_size=self.text_embedding_size, dim1=2048,dim2=256,dim3=64,dim_channel=6,name="cross_gan",splices=((self.splices-1)//2))
			self.model_crossover = self.crossover_gan.build_model()
			generator_cross = [i for i in (filter(lambda x: x.name.startswith("cross_gan_gen"),tf.trainable_variables()))]
			discriminator_cross = [i for i in (filter(lambda x: x.name.startswith("cross_gan_disc"),tf.trainable_variables()))]
			self.crossover_optimizer_g = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.model_crossover[4],var_list=generator_cross)
			self.crossover_optimizer_d = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(self.model_crossover[5],var_list=discriminator_cross) 
			self.sample_cross = self.crossover_gan.samples_generator()
	
		self.frame_gan = DCGAN(batch_size=self.batch_size,image_shape=self.video_shape, embedding_size=self.embedding_size, text_embedding_size=self.text_embedding_size, dim1=2048,dim2=256,dim3=64,dim_channel=3,name="frame_gan",splices=self.splices)
		self.model_frame = self.frame_gan.build_model()
		print("Setup model Variabless")
		generator_frame = [i for i in (filter((lambda x: x.name.startswith("frame_gan_gen")),tf.trainable_variables()))]
		discriminator_frame = [i for i in (filter(lambda x: x.name.startswith("frame_gan_disc"),tf.trainable_variables()))]

		self.frame_optimizer_g = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.model_frame[4],var_list=generator_frame)
		self.frame_optimizer_d = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.model_frame[5],var_list=discriminator_frame)
		print("Setting up Gradients")
		self.sample_frame = self.frame_gan.samples_generator()
		self.saver = tf.train.Saver()
		print("Setting up testing criterion")
		self.session = tf.InteractiveSession()

		return True

	def save_session(self,path):
		path = saver.save(self.session, path)
		print("Session saved in : " + str(path))

	def initialize(self):
		tf.global_variables_initializer().run()

	def test(self, inputs,c_inputs, ep):
		feed_dict = {
			self.sample_frame[0] : inputs[0],
			self.sample_frame[1] : inputs[1]
		}
		path = "./start_frames/sample_%d.jpg" %(ep+1)
		gen_samples = self.session.run(self.sample_frame[2],feed_dict=feed_dict)
		save_visualization(gen_samples,(32,32),path)
		if self.tree_style == 1:
			feed_dict = {
				self.sample_cross[0] : c_inputs[0],
				self.sample_cross[1] : c_inputs[1]
			}
			gen_samples = self.session.run(self.sample_cross[2],feed_dict=feed_dict)
			save_visualization(gen_samples,(32,32),path,flag=True)

	def train(self, inputs,c_inputs,tree_style=1):
		feed_dict = {
			self.model_frame[0] : inputs[0],
			self.model_frame[1] : inputs[1],
			self.model_frame[2] : inputs[2],
			self.model_frame[3] : inputs[3]
		}
		_,f_g_loss_val = self.session.run([self.frame_optimizer_g,self.model_frame[4]],feed_dict=feed_dict)
		_,f_d_loss_val = self.session.run([self.frame_optimizer_d,self.model_frame[5]],feed_dict=feed_dict)
		if self.tree_style == 1:
			feed_dict2 = {
				self.model_frame[0] : c_inputs[0],
				self.model_frame[1] : c_inputs[1],
				self.model_frame[2] : c_inputs[2],
				self.model_frame[3] : c_inputs[3]
			}
			_,c_g_loss_val = self.session.run([self.cross_optimizer_g,self.model_crossover[4]],feed_dict=feed_dict2)
			_,c_d_loss_val = self.session.run([self.cross_optimizer_d,self.model_crossover[5]],feed_dict=feed_dict2)
			return f_g_loss_val,f_d_loss_val,c_g_loss_val,c_d_loss_val
		else:
			return f_g_loss_val, f_d_loss_val

# training part
epoch = 100	
learning_rate = 1e-2
batch_size = 50
embedding_size = 128
text_embedding_size = 10

gantree = multiGAN(splices=6,video_shape=[64,64,3],tree_style=1, batch_size=batch_size, embedding_size=embedding_size, text_embedding_size=text_embedding_size,name="gantree")
gantree.build_tree()

print("Built graph for GAN, etc : ready to run")
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print(total_parameters) 

