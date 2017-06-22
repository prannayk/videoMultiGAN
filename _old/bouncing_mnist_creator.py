import numpy as np
import scipy.misc
from gensim.models import word2vec
import time

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

def convert2embedding(sentence_list,maxlen=20,batch_size = 10):
	global model
	text_embeddings = np.ndarray([batch_size, 300, maxlen])
	print(len(sentence_list))
	for i in range(len(sentence_list)):
		tokens = sentence_list[i].split()
		for j in range(len(tokens)):
			text_embeddings[i,:,j] = model[tokens[j]]
		for j in range(len(tokens),maxlen):
			text_embeddings[i,:,j] = 0
	return text_embeddings

maxlen = 20
digit_size = 28
image_size = 64
batch_size = 10
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
	flag = False
	start_time = time.time()
	while count < batch_size:
		if count % 10 == 0 and count > 0:
			if not flag:
				print("Time taken: " + str(time.time() - start_time))
				flag = True
				print("Done with %d"%(count))
				save_time = time.time()
				np.save("./bouncing_data3/image_%d.npy"%(count/10), arr=data)
#			print(data)
				np.save("./bouncing_data3/text_%d.npy"%(count/10), arr=convert2embedding(sentence_list))
				print("Saving time: " + str(time.time() - save_time))
				sentence_list = []
				data = None
				start_time = time.time()
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
		flag = False
		image = np.ndarray([1,20,64,64,1])
		img = np.ndarray([1,20,32,32,1])
		background = np.zeros([64,64,1])
		for i in range(20):
			image[0,i] = create_frame(create_frame(background,imgs[s],starts,motions,i),imgs[f],startf,motionf,i)
			img[0,i] = scipy.misc.imresize(image[0,i].reshape([64,64]),[32,32]).reshape(32,32,1)
		sentence_list.append(motion_sentence(motionf,motions,int_val(labels[f]),int_val(labels[s])))
		if data == None:
			data = img
		else:
			data = np.concatenate([data,img])
	return data,convert2embedding(sentence_list)

mnist_train_data = mnist.train.images.reshape(-1,28,28)
mnist_train_labels = mnist.train.labels

# print(np.mean(generate_gif_data(mnist_train_data, mnist_train_labels, 50)[0]))
# print(generate_gif_data(mnist_train_data,mnist_train_labels,50)[1])
print("Building training data") 
dataset = generate_gif_data(mnist_train_data, mnist_train_labels,  50000)
# print(np.mean(dataset[0]))
# print(np.mean(dataset[1]))
print("Built dataset, saving in files")
np.save("./bouncing_images.npy",arr=dataset[0])
np.save("./bouncing_sentence_embedding.npy",arr=dataset[1])
print("Saving files")
