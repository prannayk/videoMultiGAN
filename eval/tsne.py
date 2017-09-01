import scipy
import numpy as np
from sklearn.manifold import TSNE as tsne
import sys
import matplotlib.pyplot as plt
data = np.load(sys.argv[1])
with open(sys.argv[2]) as f:
	lines = f.readlines()
classes = []
for line in lines:
	classes.append(line.split())
classes = np.array(classes)[1:,1:].reshape([len(classes)-1])
print("Loaded data")
xsne = tsne(learning_rate=10).fit_transform(data)
plt.scatter(xsne[:,0], xsne[:,1],c=data)
plt.show()
