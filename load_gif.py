import sys
import subprocess

f = open("tgif-v1.0.tsv")
fdata = list()
for line in f.readlines():
	data = line.split("\t")
	title = data[0].split("/")[-1]
	name = "gif_data/" + title.split(".")[0]
	clas = title.split(".")[0].split("_")[-1]
	name = "gif_data/" + clas + "/" + title.split(".")[0]
	subprocess.call(["mkdir",name])
	subprocess.call(["wget " + data[0]],shell=True)
	subprocess.call(["mv",title, name+"/"+title])
	subprocess.call(["convert","-verbose","-coalesce",name+"/"+title,name+"/"+title.split(".")[0] + ".png"])
	fdata.append(title.split(".")[0]+"\t"+data[1])
f2 = open("data.list",mode="w")
text = ""
for t in fdata:
	text += t
f2.write(text)
print("Written text/data")
