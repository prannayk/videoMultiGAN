import tensorflow as tf
import numpy as np
import scipy.misc
from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('../google.bin', binary=True)
print("Loaded gensim")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print("Loaded MNIST")
label_dict = dict({
	1 : 'one',2 : 'two',3 : 'three',4 : 'four',5 : 'five',6 : 'six',7 : 'seven',8 : 'eight',9 : 'nine',0 : 'zero'
	})

def int_val(one_hot):
	for i in range(len(one_hot)):
		if one_hot[i] == 1:
			return i

def convert2embedding(sentence_list):
	global model,maxlen, batch_size
	text_embeddings = np.ndarray([batch_size, 300, maxlen])
	print(len(sentence_list))
	for i in range(len(sentence_list)):
		tokens = sentence_list[i].split()
		for j in range(len(tokens)):
			text_embeddings[i,:,j] = model[tokens[j]]
	return text_embeddings

maxlen = 20
digit_size = 28
image_size = 64
batch_size = 50
motions = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]

def overlap(a,b):
	return np.maximum(a,b)

def start():
	return np.random.randint(0,image_size-digit_size,2)

def motion(start):
	# print(start)
	if start[0] >= 20 and start[1] >= 20:
		return motions[np.random.randint(4,7)]
	elif start[0] >= 20 and (start[1] < 16 ):
		return motions[np.random.randint(2,5)]
	elif (start[0] < 16) and (start[1] < 16):
		return motions[np.random.randint(0,3)]
	elif (start[0] < 16) and (start[1] >= 20):
		return motions[np.random.randint(-2,1)]

def create_frame(background, overlay,start,motion, iteration):
	start_step = list(start)
	start_step[0] = start_step[0] + iteration*motion[1]
	start_step[1] = start_step[1] + iteration*motion[0]
	background[start_step[0]:start_step[0]+28,start_step[1]:start_step[1]+28,0] = (background[start_step[0]:start_step[0]+28,start_step[1]:start_step[1]+28,0] + overlay)
	return background

def motion_sentence(motionf,motions,label1, label2):
	stringval = ""
	global label_dict
	if motionf[0] == 1 :
		if motionf[1] == 1:
			stringval += ("Digit %s is moving the north west"%(label_dict[label1]))
		elif motionf[1] == 0:
			stringval += ("Digit %s is moving the north"%(label_dict[label1]))
		else:
			stringval += ("Digit %s is moving the north east"%(label_dict[label1]))
	elif motionf[0] == 0:
		if motionf[1] == 1:
			stringval += ("Digit %s is moving the west"%(label_dict[label1]))
		else:
			stringval += ("Digit %s is moving the east"%(label_dict[label1]))
	else:
		if motionf[1] == 1:
			stringval += ("Digit %s is moving the south west"%(label_dict[label1]))
		elif motionf[1] == 0:
			stringval += ("Digit %s is moving the south"%(label_dict[label1]))
		else:
			stringval += ("Digit %s is moving the east"%(label_dict[label1]))
	stringval += " while "
	if motions[0] == 1 :
		if motions[1] == 1:
			stringval += ("digit %s is moving the north west"%(label_dict[label2]))
		elif motions[1] == 0:
			stringval += ("digit %s is moving the north"%(label_dict[label2]))
		else:
			stringval += ("digit %s is moving the north east"%(label_dict[label2]))
	elif motions[0] == 0:
		if motions[1] == 1:
			stringval += ("digit %s is moving the west"%(label_dict[label2]))
		else:
			stringval += ("digit %s is moving the east"%(label_dict[label2]))
	else:
		if motions[1] == 1:
			stringval += ("digit %s is moving the south west"%(label_dict[label2]))
		elif motions[1] == 0:
			stringval += ("digit %s is moving the south"%(label_dict[label2]))
		else:
			stringval += ("digit %s is moving the east"%(label_dict[label2]))
	return stringval

def generate_gif_data(imgs,labels,batch_size):
	data = None
	sentence_list = list()
	count = 0
	while count < batch_size:
		if count % 100 == 0:
			print("Done with %d"%(count))
		# print(count+1)
		f = np.random.randint(len(imgs))
		s = np.random.randint(len(imgs))
		startf = start()
		starts = start()
		motionf = motion(startf)
		if motionf == None:
			# count -= 1
			continue
		motions = motion(starts)
		if motions == None:
			# count -= 1
			continue
		count += 1
		image = np.ndarray([1,20,64,64,1])
		background = np.zeros([64,64,1])
		for i in range(20):
			image[0,i] = create_frame(create_frame(background,imgs[s],starts,motions,i),imgs[f],startf,motionf,i)
			# print(tf.reduce_sum(image[0,i]))
			# print(tf.reduce_sum(imgs[s]))
		sentence_list.append(motion_sentence(motionf,motions,int_val(labels[f]),int_val(labels[s])))
		if data == None:
			data = image
		else:
			data = np.concatenate([data,image])
	return data,convert2embedding(sentence_list)

mnist_train_data = mnist.train.images.reshape(-1,28,28)
mnist_train_labels = mnist.train.labels

# print(np.mean(generate_gif_data(mnist_train_data, mnist_train_labels, 50)[0]))
# print(generate_gif_data(mnist_train_data,mnist_train_labels,50)[1])
print("Building training data") 
dataset = generate_gif_data(mnist_train_data, mnist_train_labels, 200000)
print("Built dataset")
start = 0
def generate_next_batch():
	global dataset, start
	start = (start+50)%200000
	data = dataset[0][start-50:start]
	sentences = dataset[1][start-50:start]
	return data, sentences
# def load_data(filename):
# 	f = open(filename, mode="r")
# 	lines = f.readlines()
# 	data = dict()
# 	images = dict()
# 	for line in lines:
# 		image = line.split("\t")[0].split("/")[0].split(".")[0]
# 		clas = image.split("_")[-1]
# 		sentence = line.split('\t')[1]
# 		data.update({sentence : ("%s/%s"%(clas,image))})
# 		image.update({sentence : image})
# 	return data, image

# data,image = load_data('tgif-v1.0.tsv')
# start = 0
# data = np.load("../bouncing_mnist_test.npy")
# def generate_batch(data,batch_size):
# 	global start
# 	relevant = data[start:start+batch_size]
# 	start+=batch_size
# 	return relevant	
	# sen_list = [i for i in np.random.permutation(list(image.keys()))]
	# start += 50
	# start = start%len(sen_list)
	# sen_list = sen_list[start:start+50]
	# video_tensor_list = list()
	# for sentence in sen_list:
	# 	gif = image[sentence]
	# 	file_list = [("./gif_data/%s/%s-%d.png"%(gif,gif,i)) for i in range(16)]
	# 	if file_list
	# 	tensor_list = list()
	# 	for file in file_list:
	# 		f = open(file)
	# 		read = f.read()
	# 		t = tf.image.decode_png(read)
	# 		l = tf.image.decode_png(tf.reshape(t, shape=([1] + t.shape)),size=[64,64])
	# 		tensor_list.append(tf.reshape(tf.reshape(l,shape=[l.shape[1],l.shape[2],l.shape[3]])),shape=[l.shape[1],l.shape[2],1,l.shape[3]])
	# 	# convert HWFC
	# 	video_tensor = tensor_list[0]
	# 	for i in range(1,len(tensor_list)):
	# 		image_tensor = tf.concat(values=[image_tensor,tensor_list[i]],axis=2)
	# 	video_tensor_list.append(image_tensor)
	# video_batch = video_tensor_list[0]
	# for i in range(1,len(video_tensor_list)):
	# 	video_batch = tf.concat(axis=0,values=[video_batch,video_tensor_list[i]])
	# return video_batch

## visualization saving
## sentence embedding

def batch_normalize(X, eps=1e-6):
	if X.get_shape().ndims == 4 :
		mean = tf.reduce_mean(X,[0,1,2])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2])
		X = (X - mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 2:
		mean = tf.reduce_mean(X,[0])
		stddev = tf.reduce_mean(tf.square(X-mean),[0])
		X = (X - mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 5:
		mean = tf.reduce_mean(X,[0,1,2,3])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2,3])	
		X = (X-mean)/tf.sqrt(stddev + eps)
	elif X.get_shape().ndims == 3:
		mean = tf.reduce_mean(X,[0,1])
		stddev = tf.reduce_mean(tf.square(X-mean),[0,1])	
		X = (X-mean)/tf.sqrt(stddev + eps)
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

class VideoGAN():
	def __init__ (self,batch_size = 50,image_shape = [64,64,1],embedding_size = 128,otext_embedding_size = 300,text_embedding_size=300,dim1 = 1024, dim2 = 128, dim3 = 64,dim4 = 16, dim_channel = 1,frames = 20,name="videogan", max_len=20):
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.embedding_size = embedding_size
		self.text_embedding_size = text_embedding_size
		self.dim1 = 2048
		self.dim2 = dim2
		self.dim3 = dim3
		self.dim_channel = dim_channel
		self.name = name
		self.frames = frames
		self.max_len = max_len
		self.otext_embedding_size = otext_embedding_size
		self.dim_4 = image_shape[0] // 4
		self.dim_2 = image_shape[0] // 2
		self.image_input_size = image_shape[0]*image_shape[1]*image_shape[2]
		self.g_weight1 = tf.Variable(tf.random_normal([embedding_size + text_embedding_size, dim1], stddev = 0.2), name=(self.name+"_generator_weight1"))
		self.g_weight2 = tf.Variable(tf.random_normal([dim1 + text_embedding_size, dim2*self.dim_4*self.dim_4], stddev = 0.2), name=(self.name+"_generator_weight2"))
		self.g_weight3 = tf.Variable(tf.random_normal([5,5,dim3,dim2+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weight3"))
		self.g_weight4 = tf.Variable(tf.random_normal([4,4,dim3*frames,dim3 + text_embedding_size],stddev=0.2),name=(self.name+"_generator_weight4"))
		self.g_weight5 = tf.Variable(tf.random_normal([5,5,dim_channel,dim3+text_embedding_size], stddev = 0.2), name=(self.name+"_generator_weight5"))

		self.d_weight1 = tf.Variable(tf.random_normal([5,5,dim_channel+text_embedding_size, dim3],stddev = 0.2), name=(self.name+"_disc_weight1"))
		self.d_weight2 = tf.Variable(tf.random_normal([4,4,frames*(dim3+text_embedding_size), dim3],stddev = 0.2), name=(self.name+"_disc_weight1"))
		self.d_weight3 = tf.Variable(tf.random_normal([5,5,dim3+text_embedding_size, dim2],stddev = 0.2), name=(self.name+"_disc_weight2"))
		self.d_weight4 = tf.Variable(tf.random_normal([dim2*self.dim_4*self.dim_4+text_embedding_size, dim1],stddev = 0.2), name=(self.name+"_disc_weight3"))
		self.d_weight5 = tf.Variable(tf.random_normal([dim1+text_embedding_size,1],stddev = 0.2), name=(self.name+"_disc_weight4"))

	def build_model(self):
		with tf.device("/gpu:0"):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			text_embedding_raw = tf.placeholder(tf.float32, [self.batch_size, self.otext_embedding_size, self.max_len])
			# text_embedding = tf.placeholder(tf.float32, [self.batch_size, self.frames, self.embedding_size])
			text_embedding = self.generate_embedding_raw(text_embedding_raw)
			r_video = tf.placeholder(tf.float32, [self.batch_size,self.frames] + self.image_shape)
			h4 = self.generate(embedding, text_embedding)
			g_video = tf.nn.sigmoid(h4)
			real_value = self.discriminate(r_video, text_embedding)
			prob_real = tf.nn.sigmoid(real_value)
			fake_value = self.discriminate(g_video, text_embedding)
			prob_fake = tf.nn.sigmoid(fake_value)
			# cost functions
			d_cost = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
			g_cost = -tf.reduce_mean(tf.log(prob_fake))
			return embedding, text_embedding, r_video, d_cost, g_cost, prob_real, prob_real

	def generate_embedding_raw(self,text_embedding):
		# naive attention
		with tf.device("/gpu:0"):
			attention = tf.Variable(tf.random_normal([self.max_len,self.frames]))
			attention_mat = tf.reshape(attention,shape=[1,self.max_len,self.frames])
			attention_matr = tf.reshape(attention,shape=[1,self.max_len,self.frames])
			for t in range(1,self.batch_size):
				attention_matr = tf.concat(values=[attention_matr,attention_mat], axis=0)
			h = batch_normalize(tf.matmul(text_embedding,attention_matr))
			return h

	def generate(self, embedding, text_embedding):
		with tf.device("/gpu:0"):
			ystack = tf.reshape(text_embedding, [self.batch_size, self.frames, 1, 1, self.text_embedding_size])
			ystack2 = tf.reshape(text_embedding[:,:,0], [self.batch_size, 1,1, self.text_embedding_size])
			embedding = tf.concat(axis=1, values=[embedding, text_embedding[:,:,0]])
			h1 = tf.nn.relu(batch_normalize(tf.matmul(embedding, self.g_weight1)))
			h1 = tf.concat(axis=1, values=[h1, text_embedding[:,:,0]])
			h2 = tf.nn.relu(batch_normalize(tf.matmul(h1,self.g_weight2)))
			h2 = tf.reshape(h2, [self.batch_size,self.dim_4,self.dim_4,self.dim2])
			h2 = tf.concat(axis=3,values=[h2,ystack2*tf.ones([self.batch_size,self.dim_4,self.dim_4,self.text_embedding_size])])

			output_shape1 = [self.batch_size,self.dim_2,self.dim_2,self.dim3]
			h3 = tf.nn.conv2d_transpose(h2,self.g_weight3,output_shape=output_shape1,strides=[1,2,2,1])
			h3 = tf.nn.relu(batch_normalize(h3))
			h3 = tf.concat(axis=3,values=[h3,ystack2*tf.ones([self.batch_size,self.dim_2,self.dim_2,self.text_embedding_size])])

			output_shape2 = [self.batch_size, self.dim_2, self.dim_2, self.dim3*self.frames]
			h4 = tf.nn.conv2d_transpose(h3,self.g_weight4, output_shape=output_shape2, strides=[1,1,1,1])
			h4 = tf.nn.relu(batch_normalize(h4))
			h5 = tf.reshape(h4, shape=[self.batch_size,self.frames ,self.dim_2, self.dim_2, self.dim3])
			h6 = tf.concat(axis=4, values=[h5,ystack*tf.ones([self.batch_size,self.frames,self.dim_2, self.dim_2, self.text_embedding_size])])

			video_shape = [self.frames] + list(self.image_shape)
			# video_shape[2] *= self.frames
			output_shape3 = [self.batch_size*self.frames] + list(self.image_shape)
			h7 = tf.reshape(h6, shape=[self.batch_size*self.frames,self.dim_2,self.dim_2,self.dim3+self.text_embedding_size])
			h8 = tf.nn.conv2d_transpose(h7, self.g_weight5, output_shape=output_shape3, strides=[1,2,2,1])
			return tf.reshape(batch_normalize(h8),shape=([self.batch_size] + video_shape))

	def discriminate(self, image, text_embedding):
		with tf.device("/gpu:0"):
			text_embedding_size = self.text_embedding_size
			height1 = self.dim_2
			height2 = self.dim_4
			ystack = tf.reshape(text_embedding[:,:,0],tf.stack([self.batch_size, 1,1, text_embedding_size]))
			ystack2 = tf.reshape(text_embedding, tf.stack([self.batch_size,self.frames,1,1, text_embedding_size]))
			video_shape = list(self.image_shape)
			video_shape[2] += 300
			image = tf.concat(axis=4, values=[image, ystack2*tf.ones([self.batch_size,self.frames] + self.image_shape)])
			image_ = tf.reshape(image, shape=([self.batch_size*self.frames] + video_shape))
			proc_image = batch_normalize(image_)
			h1 = lrelu(tf.nn.conv2d(proc_image, self.d_weight1, strides=[1,2,2,1],padding='SAME'))
			h1 = batch_normalize(h1)
			h2 = tf.reshape(h1,shape=([self.batch_size,self.frames,height1, height1, self.dim3]))
			h2 = tf.reshape(h2, shape=([self.batch_size, self.frames ,height1, height1, self.dim3]))
			h3 = tf.concat(axis=4,values=[h2, ystack2*tf.ones(shape=[self.batch_size, self.frames,height1, height1, text_embedding_size])])
			h4 = tf.reshape(h3,shape=([self.batch_size, height1, height1, self.frames*(self.dim3+text_embedding_size)]))
			h5 = lrelu(tf.nn.conv2d(h4,self.d_weight2, strides=[1,1,1,1],padding='SAME'))
			h6 = batch_normalize(tf.concat(axis=3,values=[h5, ystack*tf.ones(shape=[self.batch_size,height1, height1,text_embedding_size])]))
			h7 = tf.nn.conv2d(h6, self.d_weight3,padding='SAME', strides=[1,2,2,1])
			h8 = tf.reshape(h7, [self.batch_size,-1])
			h9 = tf.concat(axis=1,values=[h8,text_embedding[:,:,0]])
			h10 = lrelu(batch_normalize(tf.matmul(h9, self.d_weight4)))
			h11 = tf.concat(axis=1,values=[h10,text_embedding[:,:,0]])
			h12 = lrelu(batch_normalize(tf.matmul(h11, self.d_weight5)))
			return h12

	def samples_generator(self):
		with tf.device("/gpu:0"):
			batch_size = self.batch_size
			embedding = tf.placeholder(tf.float32,[batch_size, self.embedding_size])
			text_embedding = tf.placeholder(tf.float32,[batch_size,self.text_embedding_size, self.max_len])
			t = self.generate(embedding,self.generate_embedding_raw(text_embedding))
			return embedding,text_embedding,t

def save_visualization(X,ep,nh_nw=(10,20),batch_size=10, frames=20):
	Y = X.reshape(batch_size*frames, 64,64,1)
	image = np.zeros([64*nh_hw[0], w*nh_hw[1],3])
	h,w  = 64,64
	for n,x in enumerate(Y):
		j = n // nh_hw[1]
		i = n % nh_hw[1]
		image[j*h:j*h + h, i*w:i*w + w,:] = x
	scipy.misc.imsave(("bouncingmnist/sample_%d.jpg"%(ep+1)),image)


batch_size = 50
videogan = VideoGAN(batch_size=batch_size)
embedding, text_embedding, r_video, d_cost, g_cost, prob_real, prob_real = videogan.build_model()
print("Built model")

print("Built graph, exiting")

epoch = 1000
learning_rate = 1e-2

gan = VideoGAN()
embedding, sentence, real_video, d_loss, g_loss, prob_fake, prob_real = gan.build_model()
session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

g_weight_list = [i for i in filter(lambda x: x.name.startswith("videogan_gen"),tf.trainable_variables())] 
d_weight_list = [i for i in filter(lambda x: x.name.startswith("videogan_disc"), tf.trainable_variables())] 

# optimizers
with tf.device("/gpu:0"):
	g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.4).minimize(g_loss, var_list=g_weight_list)
	d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.4).minimize(d_loss, var_list=d_weight_list)

embedding_sample, sentence_sample, image_sample = gan.samples_generator()
sample_embedding, sample_text, = generate_gif_data(mnist_train_data, mnist_train_labels, 10)
tf.global_variables_initializer().run()

batch_size = 10
embedding_size = 128
text_embedding_size = 300
num_examples = 2000
epoch = 4000
print("Starting Training")
for ep in range(epoch):
	print("Running for Epoch Number: %d"%(ep+1))	
	for t in range(num_examples):
		batch,batch_text = generate_next_batch()
		random = np.random.uniform(-1,1,size=[batch_size,embedding_size]).astype(np.float32)
		feed_dict1 = {
			real_video : batch,
			embedding : random, 
			sentence : batch_text
		}
		_, g_loss_val = session.run([g_optimizer, g_loss],feed_dict=feed_dict1)
		_, d_loss_val = session.run([d_optimizer, d_loss],feed_dict=feed_dict1)
		if t%10 == 0 and t > 0:
			print("Done with batches: " + str(t*batch_size) + " Loesses :: Generator: " + str(g_loss_val) + " and Discriminator: " + str(d_loss_val) + " = " + str(d_loss_val + g_loss_val))
	print("Saving sample images and data for later testing: ")
	feed_dict = {
		embedding_sample : sample_embedding,
		sentence : sample_text,
		real_video : sample_video
	}
	gen_samples = session.run(image_sample,feed_dict=feed_dict)
	save_visualization(gen_samples, ep)
	print("Epoch: %d has been completed"%(ep + 1))
