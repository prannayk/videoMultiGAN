import os
from os.path import join, isfile
import sys
from subprocess import call


folders = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
direc = "/media/hdd/hdd/prannayk/action_reaction/"

for folder in folders : 
	path = direc + folder
	filelist = [f for f in os.listdir(path) if isfile(join(path, f))]
	for file in filelist:
		filename = file.split("/")[-1].split(".")[0]
		call(["mkdir",join(path, filename)])
		os.system("ffmpeg -i %s/%s.avi -vf fps=5 -f image2 %s-"%(path, filename, filename) + "%03d.jpg ")
		os.system("mv %s/%s-* %s/"%(path, filename, filename))
		print("Done with %s"%(filename))
