import tensorflow as tf
import numpy as np
from scipy.misc import imsave

batch_size=4
frames_input=3
frames=5

from generator import rot_text_generator as generator
def save_visualization(X, nh_nw=(1,batch_size*(frames_input+frames)), save_path='../results/%s/sample.jpg'%(sys.argv[4])):
	print(X.shape)
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

def frame_label(batch_size, frames):
	t = np.zeros([batch_size*frames, frames])
	for i in range(batch_size):
		for j in range(frames):
			t[i*frames + j,j] = 1
	return t
def morph(X):
	batch_size = int(X.shape[0])
	dim_channel = int(X.shape[-1]) // (frames_input+frames)
	print(dim_channel)
	h,w = map(lambda x: int(x), X.shape[1:3])
	img = np.zeros([(frames_input+frames)*batch_size,h,w,dim_channel])
	for i in range(batch_size):
		for t in range(frames_input+frames):
			img[i*(frames_input+frames) + t] = X[i,:,:,t*dim_channel:t*dim_channel+dim_channel]
	return img
lin="/users/gpu/prannay/mnist_stable/"
lines= []
for i in range(10):
    text="%s%05d.jpg"%(lin, i)
    lines.append(text)
    img_sample, _, img_gen, _,_ = generate(batch_size, frames, frames_input)
    save_visualization(np.concatenate([img_sample,img_gen], axis=3),save_path=text)
with open("list.txt",mode="w")  as f:
    f.write('\n'.join(lines))
