import numpy as np
import sys
import os
inputimg = np.load("%sinpimg.npy"%(sys.argv[-1]))
outputimg = np.load("%soutpimg.npy"%(sys.argv[-1]))
print(np.mean(outputimg))
diff = inputimg[:,:,:,:5] - outputimg[:,:,:,:5]
print(np.mean(diff**2))
t = np.sqrt(np.mean(diff ** 2))*255.
print(t)
