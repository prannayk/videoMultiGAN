import subprocess
path="/users/gpu/prannay/vgan/final_models/output/imgs/%s/"%(sys.argv[1])
sumval=0.
count=0.
for i in range(125):
    t = subprocess.Popen(["python","measure.py",os.path.join(path,"input_%04d"%(i+1)),os.path.join(path, "output_%04d"%(i+1))], stdout=subprocess.PIPE)
    val = t.communicate()[0].split("\n")[2].split('=')[-1]
    sumval+=val
    count+=1
print(sumval / count)
