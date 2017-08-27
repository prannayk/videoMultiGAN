import os
import numpy as np
from os.path import join, isfile
import sys
from subprocess import call
from PIL import Image

folders = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
direc = "/home/accents/video_source/"

for folder in folders : 
	path = direc
	filelist = [f for f in os.listdir(path) if isfile(join(path, f)) and (folder in f) and (not 'npy' in f)]
	for file in filelist:
		filename = file.split("/")[-1].split(".")[0]
		call(["mkdir",join(path, filename)])
		os.system("ffmpeg -i %s/%s.avi -vf fps=25 -s 40x32 -f image2 %s/%s/%s-"%(path, filename, path, filename, filename) + "%03d.png ")
		os.system("mv %s/%s-* %s/"%(path, filename, filename))
		print("Done with %s"%(filename))
		path_file = direc + "/" + filename
		images =[f for f in os.listdir(path_file) if isfile(join(path_file, f))]
		images = images[:90]
		frames = np.zeros([90, 32, 40, 3])
		for i,img in enumerate(images) : 
			im = np.array(Image.open("%s/%s/%s"%(path, filename, img)).getdata())
			print(np.sum(im))
			print(np.mean(im))
			if (im.shape != [16384,3]):
				continue
			im = (im / 255. ).reshape([128,128, 3])
			frames[i] = im
		np.save("/mnt/video_small/video_25_%s.npy"%(filename), frames)
		os.system("rm -rf %s/%s")
