import os
import numpy as np
from os.path import join, isfile
import sys
from subprocess import call
from PIL import Image

folders = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
direc = "/media/hdd/hdd/prannayk/action_reaction/"
num=int(sys.argv[1])
for folder in folders : 
	path = direc + folder
	filelist = [f for f in os.listdir(path) if isfile(join(path, f))]
	for file in filelist:
		filename = file.split("/")[-1].split(".")[0]
		call(["mkdir",join(path, filename)])
		os.system("ffmpeg -i %s/%s.avi -vf fps=%d -s 128x128 -f image2 %s/%s/%s-"%(path, filename, num, path, filename, filename) + "%03d.png ")
		os.system("mv %s/%s-* %s/"%(path, filename, filename))
		print("Done with %s"%(filename))
		path_file = direc + folder + "/" + filename
		images =[f for f in os.listdir(path_file) if isfile(join(path_file, f))]
		images = images[:64]
		frames = np.zeros([64, 128, 128, 3])
		for i,img in enumerate(images) : 
			im = np.array(Image.open("%s/%s/%s"%(path, filename, img)).getdata())
			print(im.shape)
			im = (im / 255. ).reshape([128,128, 3])
			frames[i] = im
		np.save("%s/video_%d_%s.npy"%(direc, num, filename), frames)
		os.system("rm -rf %s/%s")
