import tensorflow as tf
import numpy as np
import scipy.misc
import time

start = 0
current = 0
num = 100

def generate_next_batch(frames,batch_size,start, current):
	global images_train, text_train,num
	t = (5*num) // batch_size
	if current >= t or start == 0:
		images_train, text_train, start, current = load_batches(num, batch_size, start, current)
		current = 0
	images = images_train[current*batch_size:current*batch_size + batch_size]
	text = text_train[current*batch_size:current*batch_size + batch_size]
	current += 1
	return images, text, start, current

def load_batches(num,batch_size,start, current):
	image = "bouncing_data/image_%d.npy"%(start+1)
	text_file = "bouncing_data/text_%d.npy"%(start+1)
	t = np.load(text_file)
	img = np.load(image)
	for i in range(num-1):
		image = "bouncing_data/image_%d.npy"%(start+i+2)
		print(image)
		text_file = "bouncing_data/text_%d.npy"%(start+i+2)
		t2 = np.load(text_file)
		im2 = np.load(image)
		img = np.concatenate([img,im2],axis=0)
		t = np.concatenate([t,t2],axis=0)
	start += num
	if start > 50000:
		start = 0
		current = 0
	return img, t, start,current

def save_visualization(X,ep,nh_nw=(20,100),batch_size = 100, frames=20):
	h,w = 64,64
	Y = X.reshape(batch_size*frames, h,w,1)
	image = np.zeros([h*nh_nw[0], w*nh_nw[1],3])
	for n,x in enumerate(Y):
		j = n // nh_nw[1]
		i = n % nh_nw[1]
		image[j*h:j*h + h, i*w:i*w + w,:] = x
	scipy.misc.imsave(("bouncingsample/sample_%d.jpg"%(ep+1)),image)


batch_size = 100
print("Built model")

epoch = 1000
learning_rate = 1e-3

embedding_size = 96
text_embedding_size = 150
num_examples = 500
epoch = 200
frames = 20

sample_video, sample_text,start, current = generate_next_batch(20,batch_size,start,current)
print("Saving")
save_visualization(sample_video,-1)
