import numpy as np
from scipy.misc import imsave
rname = sys.argv[-1]
inputimgs = np.load("%sinputimg.npy"%(rname))
outputimgs = np.load("%soutputimg.npy"%(rname))
def save_visualization(X, nh_nw=(batch_size,frames_input+frames), save_path='../../results/%s/sample.jpg'%(sys.argv[4])):
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

i=0
while i < inputimgs.shape[0]:
    save
