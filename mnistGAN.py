import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

graph = tf.Graph()

batch_size = 50

with graph.as_default():
	g_input = tf.placeholder(tf.float32,shape=(batch_size,embedding_size))
	