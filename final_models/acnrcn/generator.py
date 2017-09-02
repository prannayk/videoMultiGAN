from PIL import Image
import numpy as np
import os

lf = list()

def load_path(num):
    with open("path.txt") as f :
        line = f.readlines()
        line = filter(lambda x : x != '\n', line[num])
    return line

def video_loader(num):
	global lf
	path = load_path(num)
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith("video") and ('npy' in f)]
	lf = files

def video_next_batch(batch_size, frames, frames_input=2):
	global lf
	if len(lf) == 0 :
		video_loader(0)
	path = load_path(0)
	file_list = np.random.choice(lf, batch_size , replace=True)
	video_list = np.zeros([batch_size, 30, 32, 40, 3])
	for i,fil in enumerate(file_list):
		complete_path = path + fil
		videonpy = np.load(complete_path)
		video_list[i] = videonpy[:30]
	start_num = [i for i in range(30-(frames+2))]
	start_list = np.random.choice(start_num, batch_size, replace=True)
	video_batch = np.zeros([batch_size, 32, 40, frames + frames_input])
	motion_class = []
	person_class = []
	style_class = []
	frame_speed = []
	for i in range(batch_size):
		for j in range(frames+frames_input):
			video_batch[i,:,:,j] = video_list[i,start_list[i]+j:start_list[i]+j+1,:,:,0] / 255.
		list_feat = file_list[i].split(".")[0].split("_")[1:5]
		if len(list_feat) < 3 : 
			print(list_feat)
		if list_feat[2] == "walking" :
			motion_class.append(5)
		elif list_feat[2] == "running" :
			motion_class.append(4)
		elif list_feat[2] == "jogging" :
			motion_class.append(3)
		elif list_feat[2] == "handclapping" : 
			motion_class.append(2)
		elif list_feat[2] == "handwaving" : 
			motion_class.append(1)
		else:
			motion_class.append(0)
		person_class.append(int(list_feat[1][6:]) - 1)
		style_class.append(int(list_feat[3][1:]) - 1)
		frame_speed.append(list_feat[0])
	person_hot = np.zeros([batch_size, 25])
	style_hot = np.zeros([batch_size, 4])
	motion_hot = np.zeros([batch_size, 7])
	person_hot[np.arange(batch_size), person_class] = 1
	style_hot[np.arange(batch_size), style_class] = 1
	motion_hot[np.arange(batch_size), motion_class] = 1
	motion_hot[:,-1] = frame_speed
	return video_batch[:,:,:,:frames_input], video_batch[:,:,:,frames_input-1:frames_input-1+frames], video_batch[:,:,:,frames_input:], person_hot, motion_hot, style_hot

def video_large_batch(batch_size, frames, frames_input=2):
	global lf
	if len(lf) == 0 :
		video_loader(1)
	path = load_path(1)
	file_list = np.random.choice(lf, batch_size , replace=True)
	video_list = np.zeros([batch_size, 30, 64, 64, 3])
	for i,fil in enumerate(file_list):
		complete_path = path + fil
		videonpy = np.load(complete_path)
		video_list[i] = videonpy[:30]
	start_num = [i for i in range(30-(frames+2))]
	start_list = np.random.choice(start_num, batch_size, replace=True)
	video_batch = np.zeros([batch_size, 64, 64, frames + frames_input])
	motion_class = []
	person_class = []
	style_class = []
	frame_speed = []
	for i in range(batch_size):
		for j in range(frames+frames_input):
			video_batch[i,:,:,j] = video_list[i,start_list[i]+j:start_list[i]+j+1,:,:,0] / 255.
		list_feat = file_list[i].split(".")[0].split("_")[1:5]
		if len(list_feat) < 3 : 
			print(list_feat)
		if list_feat[2] == "walking" :
			motion_class.append(5)
		elif list_feat[2] == "running" :
			motion_class.append(4)
		elif list_feat[2] == "jogging" :
			motion_class.append(3)
		elif list_feat[2] == "handclapping" : 
			motion_class.append(2)
		elif list_feat[2] == "handwaving" : 
			motion_class.append(1)
		else:
			motion_class.append(0)
		person_class.append(int(list_feat[1][6:]) - 1)
		style_class.append(int(list_feat[3][1:]) - 1)
		frame_speed.append(list_feat[0])
	person_hot = np.zeros([batch_size, 25])
	style_hot = np.zeros([batch_size, 4])
	motion_hot = np.zeros([batch_size, 7])
	person_hot[np.arange(batch_size), person_class] = 1
	style_hot[np.arange(batch_size), style_class] = 1
	motion_hot[np.arange(batch_size), motion_class] = 1
	motion_hot[:,-1] = frame_speed
	return video_batch[:,:,:,:frames_input], video_batch[:,:,:,frames_input-1:frames_input-1+frames], video_batch[:,:,:,frames_input:], person_hot, motion_hot, style_hot



def rot_generator(batch_size, frames,frames_input=2):
	return video_next_batch(batch_size, frames, frames_input=frames_input)

def large_generator(batch_size, frames,frames_input=2):
	return video_large_batch(batch_size, frames, frames_input=frames_input)

def validation_generator(batch_size, frames, frames_input=3):
	feed_list = video_next_batch(batch_size, frames, frames_input=frames_input)
	return feed_list[0], feed_list[1], feed_list[0], feed_list[3], feed_list[4], feed_list[5]

word_len = 14

def sentence_proc(one_hot, rot):
	num_dict = {
		0 : "zero",
		1 : "one",
		2 : "two",
		3 : "three",
		4 : "four",
		5 : "five",
		6 : "six",
		7 : "seven",
		8 : "eight",
		9 : "nine"
	}
	for i in range(len(one_hot)):
		if one_hot[i] == 1:
			string1 = num_dict[i]
	if rot > 0 : 
		string2 = "clockwise"
	else :
		string2 = "anti clockwise"
	return string1, string2

def convert_embedding(sentence):
	global dictionary
	embed = np.array(map(lambda x: dictionary[x], sentence.split()))
	t = np.zeros([word_len,300])
	t[:len(embed)] = embed
	del embed
	return t


def rot_text_generator(batch_size, frames):
	global word_len
	batch1in, batch1_labels = video_next_batch(batch_size)
	batch1 = batch1in.reshape([batch_size, 28,28])
	batch2 = batch1in.reshape([batch_size, 28,28])
	batch = np.zeros([batch_size, 64, 64,9])
	batch_gen = np.zeros([batch_size, 64, 64,3*frames])
	batch_labels = np.zeros([batch_size, 13])
	batch_labels[:,:10] += batch1_labels
	text_labels = np.zeros([batch_size, word_len, 300])
	for i in range(batch_size):
		t = np.random.randint(0,32 // (frames+2) + 1)
		l = np.random.randint(0,256,[3]).astype(float) / 255
		batch_labels[i,10:] = l
		random = np.random.randint(0,4)
		rot = np.random.normal(0,5)
		if t == 0:
			sentence = "the digit %s is moving to the left downwards while it rotates %s"%(sentence_proc(batch1_labels[i], rot))
			text_labels[i] = convert_embedding(sentence)
			# text_labels[i] = np.array([rot,-1,1,1,-1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(3):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch2[i]*l[j]
		elif t==1 :
			sentence = "the digit %s is moving to the right downwards while it rotates %s"%(sentence_proc(batch1_labels[i], rot))
			text_labels[i] = convert_embedding(sentence)
			# text_labels[i] = np.array([rot, 1,-1,-1,1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(3):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch2[i]*l[j]
		elif t==2 :
			sentence = "the digit %s is moving to the right upwards while it rotates %s"%(sentence_proc(batch1_labels[i], rot))
			text_labels[i] = convert_embedding(sentence)
			# text_labels[i] = np.array([rot, -1,-1,1,1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(3):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch2[i]*l[j]
		else :
			sentence = "the digit %s is moving to the right upwards while it rotates %s"%(sentence_proc(batch1_labels[i], rot))
			text_labels[i] = convert_embedding(sentence)
			# text_labels[i] = np.array([rot, 1,1,-1,-1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(3):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate(r*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch2[i]*l[j]
			for r in range(frames):
				batch2[i] = (np.array(Image.fromarray(batch1[i] * 255.).rotate((j+3)*rot, Image.BILINEAR).getdata()) / 255.).reshape(28,28)
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch2[i]*l[j]
	return batch, batch_gen, batch_labels, text_labels
def text_generator(batch_size):
	batch1, batch1_labels = video_next_batch(batch_size)
	batch1 = batch1.reshape([batch_size, 28,28])
	batch = np.zeros([batch_size, 64, 64,6])
	batch_gen = np.zeros([batch_size, 64, 64,3*frames])
	batch_labels = np.zeros([batch_size, 13])
	batch_labels[:,:10] += batch1_labels
	# text_labels = np.zeros([batch_size, 15])
	for i in range(batch_size):
		t = np.random.randint(0,32 // (frames+2) + 1)
		l = np.random.randint(0,256,[3]).astype(float) / 255
		batch_labels[i,10:] = l
		random = np.random.randint(0,5)
		if t == 0:
			# text_labels[i] = np.array([-1,1,1,-1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch1[i]*l[j]
		elif t==1 :
			# text_labels[i] = np.array([1,-1,-1,1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch1[i]*l[j]
		elif t==2 :
			# text_labels[i] = np.array([-1,-1,1,1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,34-(random*r):62-(random*r),2+(random*r):30+(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 26-(random*r):54-(random*r),10+(random*r):38+(random*r),j+(3*r)] = batch1[i]*l[j]
		else :
			# text_labels[i] = np.array([1,1,-1,-1])
			# text_labels[i][-1] *= random
			# text_labels[i][-2] *= random
			for r in range(2):
				for j in range(3):
					batch[i,2+(random*r):30+(random*r),34-(random*r):62-(random*r),j+(3*r)] = batch1[i]*l[j]
			for r in range(frames):
				for j in range(3):
					batch_gen[i, 10+(random*r):38+(random*r),26-(random*r):54-(random*r),j+(3*r)] = batch1[i]*l[j]
	return batch, batch_gen, batch_labels, # text_labels
