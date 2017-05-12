import tensorflow as tf
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def weight_variable(shape,stddev):
	x = tf.truncated_normal(shape,stddev=stddev)
	return tf.Variable(x)

def bias_variable(shape):
	x = tf.zeros(shape)
	return tf.Variable(x)

print("Building Graph")
train_input = tf.placeholder(tf.float32,shape=[None, 784])
train_output = tf.placeholder(tf.float32, shape=[None, 10])
dropout = tf.placeholder(tf.float32)

weight1 = weight_variable([5,5,1,8],1.0)
bias1 = bias_variable([8])

train_image = tf.reshape(train_input, [-1,28,28,1])

hidden1 = tf.nn.relu(tf.nn.conv2d(train_image,weight1,strides=[1,1,1,1],padding='SAME') + bias1)
hiddenp1 = tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

weight2 = weight_variable([5,5,8,32],0.5)
bias2 = bias_variable([32])

hidden2 = tf.nn.relu(tf.nn.conv2d(hiddenp1,weight2,strides=[1,1,1,1],padding='SAME') + bias2)
hiddenp2 = tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

weight5 = weight_variable([2,2,32,64],0.5)
bias5 = bias_variable([64])

hiddenp3 = tf.nn.relu(tf.nn.conv2d(hiddenp2,weight5,strides=[1,1,1,1],padding='SAME') + bias5)

weight3 = weight_variable([7*7*64,1024],1.0)
bias3 = bias_variable([1024])

hidden3 = tf.reshape(hiddenp3,[-1,7*7*64])
hidden4 = tf.nn.relu(tf.matmul(hidden3,weight3) + bias3)

hidden_drop = tf.nn.dropout(hidden4,dropout)

weight4 = weight_variable([1024,120],1.0)
weight6 = weight_variable([120,10],1.0)
bias4 = bias_variable([120])
bias6 = bias_variable([10])

hidden5 = tf.matmul(hidden_drop,weight4) + bias4
hidden6 = tf.matmul(hidden5,weight6) + bias6

output = tf.nn.softmax(hidden6)
learning_rate = tf.placeholder(tf.float32)
cross_entropy = -tf.reduce_sum(train_output*tf.log(output))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(train_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
init = tf.initialize_all_variables()

num_steps = 20000
max_accuracy = 0
count = 0
avg_accuracy = 0
print("Running session:")
rate = 1e-4
session = tf.InteractiveSession()
session.run(init)
iterations = 5 
for i in range(num_steps):
	batch = mnist.train.next_batch(50)
	feed_dict = {
		train_input : batch[0],
		train_output : batch[1],
		dropout : 0.8,
		learning_rate : rate
	}
	for j in range(iterations):
		train_step.run(feed_dict=feed_dict)
	if i%50 == 0 and i > 0:
		feed_dict[dropout] = 1
		train_accuracy = accuracy.eval(feed_dict=feed_dict)
		avg_accuracy += train_accuracy
		if(train_accuracy > max_accuracy) :
			max_accuracy = train_accuracy
			saver.save(session, './mnist_best.ckpt')
		if max_accuracy > 0.92:
			break
		print("Step " + str(i) + " and accuracy : " + str(train_accuracy) + " and count : " + str(count) + "/50")
	if i%500 == 0:
		avg_accuracy /= 10
		print("Average accuracy is : " + str(avg_accuracy))
		avg_accuracy = 0
	if i%1000 == 0 and i > 0:
		rate *= 0.8
feed_dict = {
	train_input : mnist.test.images,
	train_output : mnist.test.labels,
	dropout : 1
}
print("Final accuracy : " + str(accuracy.eval(feed_dict=feed_dict)))
path = saver.save(session, './mnist.ckpt')
print("Saved in file with name as : "  + str(path))
