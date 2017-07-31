import os
from os.path import join, isfile
import sys
from subprocess import call


folders = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
direc = "/media/hdd/hdd/prannayk/action_reaction/"

for folder in folders[:1] : 
	path = direc + folder
	filelist = [f for f in os.listdir(path) if isfile(join(path, f))]
	for file in filelist[:10]:
		filename = file.split("/")[-1].split(".")[0]
		call(["mkdir",join(path, filename)])
		os.system("ffmpeg -i %s/%s.avi -vf fps=5 -f image2 %s/%s/%s-"%(path, filename, path, filename, filename) + "%03d.png ")
		os.system("mv %s/%s-* %s/"%(path, filename, filename))
		print("Done with %s"%(filename))
