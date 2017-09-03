import numpy as np
import sys
import os
inputimg = np.load("%sinputimg.npy"%(sys.argv[-1]))
outputimg = np.load("%soutputimg.npy"%(sys.argv[-1]))
diff = inputimg[:,:,:,:5] - outputimg[:,:,:,:5]
t = np.sqrt(np.mean(diff ** 2))*255.
print(t)
