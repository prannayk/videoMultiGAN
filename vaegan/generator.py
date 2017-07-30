from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/hdd/hdd/data_backup/prannayk/MNIST_data/", one_hot=True)
from PIL import Image
import numpy as np
def rot_generator(batch_size, frames):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28,28])
	batch = np.zeros([batch_size, 64, 64,9])
	batch_gen = np.zeros([batch_size, 64, 64,3*frames])
	batch_labels = np.zeros([batch_size, 13])
	batch_labels[:,:10] += batch1_labels
	text_labels = np.zeros([batch_size, 5])
	for i in range(batch_size):
		t = np.random.randint(0,32 // (frames+2) + 1)
		l = np.random.randint(0,256,[3]).astype(float) / 255
		batch_labels[i,10:] = l
		random = np.random.randint(0,5)
		rot = np.random.normal(0,5)
		if t == 0:
			text_labels[i] = np.array([rot,-1,1,1,-1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(3):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch2[i]*l[j]
		elif t==1 :
			text_labels[i] = np.array([rot, 1,-1,-1,1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch2[i]*l[j]
		elif t==2 :
			text_labels[i] = np.array([rot, -1,-1,1,1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch2[i]*l[j]
		else :
			text_labels[i] = np.array([rot, 1,1,-1,-1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata() / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch2[i]*l[j]
	return batch, batch_gen, batch_labels, text_labels

def text_generator(batch_size):
	batch1, batch1_labels = mnist.train.next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28,28])
	batch = np.zeros([batch_size, 64, 64,6])
	batch_gen = np.zeros([batch_size, 64, 64,3*frames])
	batch_labels = np.zeros([batch_size, 13])
	batch_labels[:,:10] += batch1_labels
	text_labels = np.zeros([batch_size, 15])
	for i in range(batch_size):
		t = np.random.randint(0,32 // (frames+2) + 1)
		l = np.random.randint(0,256,[3]).astype(float) / 255
		batch_labels[i,10:] = l
		random = np.random.randint(0,5)
		if t == 0:
			text_labels[i] = np.array([-1,1,1,-1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch1[i]*l[j]
		elif t==1 :
			text_labels[i] = np.array([1,-1,-1,1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch1[i]*l[j]
		elif t==2 :
			text_labels[i] = np.array([-1,-1,1,1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch1[i]*l[j]
		else :
			text_labels[i] = np.array([1,1,-1,-1])
			text_labels[i][-1] *= random
			text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch1[i]*l[j]
	return batch, batch_gen, batch_labels, text_labels
