import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, 
	_kl_divergence)
from sklearn.utils.extmath import _ravel

RS = 20170829
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from optparse import OptionParser
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
	rc={"lines.linewidth": 2.5})

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
parser = OptionParser()
parser.add_option("-i","--input", dest="output",
	help="Output File", metavar="FILE")
parser.add_option("-c","--classes",dest="classes",
	help="Number of classes", metavar="FILE")
(options, args) = parser.parse_args()
data = np.load(options["output"])
size = int(options["classes"])

def scatter(x, colors):
	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", size))

	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		c=palette[colors.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	# We add the labels for each digit.
	txts = []
	for i in range(10):
		# Position of each label.
		xtext, ytext = np.median(x[colors == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=24)
		txt.set_path_effects([
			PathEffects.Stroke(linewidth=5, foreground="w"),
			PathEffects.Normal()])
		txts.append(txt)

	return f, ax, sc, txts
x = np.vstack([data.data[data.target==i] for i in range(size)])
y = np.hstack([data.target[data.target==i] for i in range(size)])
digits_proj = TSNE(random_state=RS).fit_transform(X)
scatter(digits_proj, y)
plt.savefig('images/%s_tsne-generated.png'%(options["output"]), dpi=120)