import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from generator import rot_generator as generate_rot
from generator import text_generator as generate_text

def create_dataset(batch_size, total_size, image_input_shape, image_create_shape, frames,type="rot"):
    image_start = np.zeros(shape=[total_size] + image_input_shape)
    image_gen = np.zeros(shape=[total_size] + image_create_shape)
    image_labels = np.zeros(shape=[total_size, 13])
    image_motion_labels = np.zeros(shape=[total_size, 4])
    for i in range(total_size // batch_size):
        if i % 1000 == 0:
            print(i)
        if type == "rot":
            output_list = generate_rot(batch_size, frames)
        elif type == "text":
            output_list = generate_text(batch_size, frames)
        else :
            raise NotImplemented
        image_start[i*batch_size : i*batch_size + batch_size] = output_list[0]
        image_gen[i*batch_size:i*batch_size + batch_size] = output_list[1]
        image_labels[i*batch_size:i*batch_size + batch_size] = output_list[2]
        image_motion_labels[i*batch_size:i*batch_size + batch_size] = output_list[3]
    dataset = {
        "image_start" : image_start,
        "image_gen" : image_gen,
        "image_labels" : image_labels,
        "image_motion_labels" : image_motion_labels
    }
    return dataset
frames=6
dataset = create_dataset(batch_size=64 // frames, total_size=64000, image_input_shape=[64,64,6], image_create_shape=[64,64,3*frames], frames=frames)

np.save("/media/hdd/hdd/prannayk/mnist_data/dataset_image_start_%d_%d_%d_2_%d.npy"%(64, 64000,64 // frames, frames), dataset["image_start"])
np.save("/media/hdd/hdd/prannayk/mnist_data/dataset_image_gen_%d_%d_%d_2_%d.npy"%(64, 64000,64 // frames, frames), dataset["image_gen"])
np.save("/media/hdd/hdd/prannayk/mnist_data/dataset_image_labels_%d_%d_%d_2_%d.npy"%(64, 64000,64 // frames, frames), dataset["image_labels"])
np.save("/media/hdd/hdd/prannayk/mnist_data/dataset_image_motion_labels_%d_%d_%d_2_%d.npy"%(64, 64000,64 // frames, frames), dataset["image_motion_labels"])
