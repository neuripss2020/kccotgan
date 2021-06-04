#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import cv2

from scipy.stats import multivariate_normal
from functools import partial
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import rbf_kernel
import random

import collections
import os
import tensorflow as tf
from matplotlib import animation, rc
from IPython.display import HTML
nest = tf.nest

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 7,
        }


class DataProcessor:
    def __init__(self, path, seq_len, channels):
        self.training_path = path
        self.sequence_length = seq_len
        self.channels = channels

    def get_dataset_from_path(self, buffer):
        read_data = tf.data.Dataset.list_files(self.training_path)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_video_data(self, buffer, batch_size, height, width):
        '''
        :return: tf dataset
        '''
        def read_tfrecord(serialized_example):
            features = {'x': tf.io.FixedLenFeature([height * width * self.sequence_length * self.channels, ],
                                                   dtype=tf.float32)}
            example = tf.io.parse_single_example(serialized_example, features)
            return example['x']

        dataset = self.get_dataset_from_path(buffer)
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        return dataset


def load_kth_data(batch_size, height=120, width=160, time_step=16, path='../../data/kth'):
    list_f = [x for x in os.listdir(path)]

    for x in range(batch_size):
        rand_folder = random.choice(list_f)
        path_to_file = path + '/' + rand_folder
        file_name = random.choice(os.listdir(path_to_file))
        path_to_video = path_to_file + '/' + file_name
        vidcap = cv2.VideoCapture(path_to_video)
        n_frames = vidcap.get(7)
        frame_rate = vidcap.get(5)
        ret, frame = vidcap.read()
        # print(n_frames, rand_folder, file_name, frame_rate, frame.shape)
        stacked_frames = []
        while vidcap.isOpened():
            frame_id = vidcap.get(1)  # current frame number
            ret, frame = vidcap.read()
            if not ret or len(stacked_frames) > (time_step - 1):
                break
            frame = frame / 255.0
            if rand_folder == 'running' or rand_folder == 'walking' or rand_folder == 'jogging':
                if frame_id % 1 == 0 and frame_id > 5:
                    resized_frame = tf.image.resize(frame, size=[height, width], method='nearest')
                    cropped_frame = tf.image.resize_with_crop_or_pad(resized_frame, height, width)
                    stacked_frames.append(cropped_frame)
            elif n_frames < 350:
                if frame_id % 1 == 0 and frame_id > 5:
                    resized_frame = tf.image.resize(frame, size=[height, width], method='nearest')
                    cropped_frame = tf.image.resize_with_crop_or_pad(resized_frame, height, width)
                    stacked_frames.append(cropped_frame)
            else:
                if frame_id % 1 == 0 and frame_id > 10:
                    resized_frame = tf.image.resize(frame, size=[height, width], method='nearest')
                    cropped_frame = tf.image.resize_with_crop_or_pad(resized_frame, height, width)
                    stacked_frames.append(cropped_frame)

            if len(stacked_frames) < time_step:
                continue

        stacked_frames = np.reshape(stacked_frames, newshape=(time_step, height, width, 3))
        stacked_frames = np.transpose(stacked_frames, (1, 0, 2, 3))
        stacked_frames = np.reshape(stacked_frames, newshape=(height, time_step * width, 3))
        yield stacked_frames


def display_images(x, row, col, batch_size, height, width, iters, saved_file):
    fig, axe = plt.subplots(row, col, figsize=(8, 8))

    for i in range(row):
        for j in range(col):
            axe[i][j].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width]), origin="upper",
                             cmap="gray", interpolation="nearest")
            axe[i][j].set_xticks([])
            axe[i][j].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def display_frames(x, row, batch_size, seq_len, height, width, channels, iters, saved_file):
    fig, axe = plt.subplots(row, figsize=(8, 8))

    for i in range(row):
        if channels > 1:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len, channels]),
                          origin="upper", cmap="gray", interpolation="nearest")
        else:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len]),
                          origin="upper", cmap="gray", interpolation="nearest")
        axe[i].set_xticks([])
        axe[i].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def check_model_summary(batch_size, seq_len, model, stateful=False):
    if stateful:
        inputs = tf.keras.Input((batch_size, seq_len))
    else:
        inputs = tf.keras.Input((batch_size, seq_len))
    outputs = model.call(inputs)

    model_build = tf.keras.Model(inputs, outputs)
    print(model_build.summary())

'''
Modified code originally obtained from https://github.com/deepmind/gqn-datasets 
'''

class Quantiser:
    def __init__(self, n=50, codebook=None):
        self.n_blocks = n
        self.codebook = codebook

        if self.codebook is None:
            self.codebook = self.create_codebook()

    def create_codebook(self, left_lim=0.0, right_lim=1.0):
        edge_len = (right_lim - left_lim) / tf.cast(self.n_blocks, tf.float32)
        limits = tf.range(left_lim, right_lim+1e-06, delta=edge_len, dtype=tf.float32)
        centroids = (limits[1:] + limits[:-1]) / 2
        return centroids

    def quantise(self, inputs):
        '''
        Args:
            inputs: inputs that has shape [batch_size, height, timesteps, width, channels]
            or [batch_size, timesteps, channels]
        Returns:
            quantized output with the same shape as inputs
        '''
        if len(inputs.shape) == 5:
            #batch_size, h, time_steps, w, nchannel = tf.shape(inputs)
            #print(inputs._tf_api_names)
            inputs = tf.transpose(inputs, (0, 2, 1, 3, 4))
            #print(inputs._tf_api_names)
            inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1],
                                         tf.shape(inputs)[2] * tf.shape(inputs)[3], tf.shape(inputs)[4]])
            #print(inputs._tf_api_names)
            codes = tf.tile(self.codebook[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis],
                            [1, tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2],  tf.shape(inputs)[3]])
            distance = (inputs[tf.newaxis, :, :] - codes) ** 2
            indices = tf.argmin(distance, 0)
            quantized = tf.nn.embedding_lookup(self.codebook, indices)
            quantized = tf.reshape(quantized, [tf.shape(inputs)[0], tf.shape(inputs)[1], 64, 64, tf.shape(inputs)[3]])
            quantized = tf.transpose(quantized, (0, 2, 1, 3, 4))
        elif len(inputs.shape) == 3:
            batch_size, time_steps, nchannel = tf.shape(inputs)
            codes = tf.tile(self.codebook[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, batch_size, time_steps, nchannel])
            distance = (inputs[tf.newaxis, :, :] - codes) ** 2
            indices = tf.argmin(distance, 0)
            quantized = tf.nn.embedding_lookup(self.codebook, indices)
            quantized = tf.reshape(quantized, [batch_size, time_steps, nchannel])
        else:
            raise RuntimeError("Shape not correct.")
        return quantized


"""Minimal data reader for GQN TFRecord datasets."""

# nest = tf.contrib.framework.nest
nest = tf.nest
seed = 1

DatasetInfo = collections.namedtuple('DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_size',
                                                     'sequence_size'])
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == 'train':
        num_files = dateset_info.train_size
    else:
        num_files = dateset_info.test_size

    length = len(str(num_files))
    template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
    return [os.path.join(base, template.format(i + 1, num_files))
            for i in range(num_files)]


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class DataReader(object):
    """Minimal queue based TFRecord reader.
      You can use this reader to load the datasets used to train Generative Query
      Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
      See README.md for a description of the datasets and an example of how to use
      the reader.
      """

    def __init__(self,
                 dataset,
                 time_steps,
                 root,
                 mode='train',
                 # Optionally reshape frames
                 custom_frame_size=None):
        """Instantiates a DataReader object and sets up queues for data reading.
        Args:
          dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
              'rooms_free_camera_no_object_rotations',
              'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
              'shepard_metzler_7_parts'].
          time_steps: integer, number of views to be used to assemble the context.
          root: string, path to the root folder of the data.
          mode: (optional) string, one of ['train', 'test'].
          custom_frame_size: (optional) integer, required size of the returned
              frames, defaults to None.
          num_threads: (optional) integer, number of threads used to feed the reader
              queues, defaults to 4.
          capacity: (optional) integer, capacity of the underlying
              RandomShuffleQueue, defualts to 256.
          min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
              RandomShuffleQueue, defualts to 128.
          seed: (optional) integer, seed for the random number generators used in
              the reader.
        Raises:
          ValueError: if the required version does not exist; if the required mode
             is not supported; if the requested time_steps is bigger than the
             maximum supported for the given dataset version.
        """

        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, _DATASETS.keys()))

        if mode not in _MODES:
            raise ValueError('Unsupported mode {} requested. Supported modes '
                             'are {}'.format(mode, _MODES))

        self._dataset_info = _DATASETS[dataset]

        if time_steps > self._dataset_info.sequence_size:
            raise ValueError(
                'Maximum support context size for dataset {} is {}, but '
                'was {}.'.format(
                    dataset, self._dataset_info.sequence_size, time_steps))

        self.time_steps = time_steps
        self._custom_frame_size = custom_frame_size

        with tf.device('/cpu'):
            self._queue = _get_dataset_files(self._dataset_info, mode, root)

    def get_dataset_from_path(self, buffer=100):
        read_data = tf.data.Dataset.list_files(self._queue)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_dataset(self, batch_size):
        """Instantiates the ops used to read and parse the data into tensors."""
        def read_tfrecord(serialized_example):
            feature_map = {'frames': tf.io.FixedLenFeature(shape=self._dataset_info.sequence_size, dtype=tf.string)}
            example = tf.io.parse_example(serialized_example, feature_map)
            frames = self._preprocess_frames(example)
            return frames

        dataset = self.get_dataset_from_path()
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        return dataset

    def _preprocess_frames(self, example):
        """Instantiates the ops used to preprocess the frames data."""
        frames = tf.concat(example['frames'], axis=0)
        frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(_convert_frame_data,  tf.reshape(frames, [-1]),
                                                                   dtype=tf.float32))
        dataset_image_dimensions = tuple([self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
        frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + dataset_image_dimensions)
        if (self._custom_frame_size and
                self._custom_frame_size != self._dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(frames, new_frame_dimensions[:2])
            frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + new_frame_dimensions)
        return tf.transpose(tf.squeeze(frames[:, :self.time_steps, :, :]), (1, 0, 2, 3))


def samples_to_video(samples, nx, ny, time_steps=16, x_height=64, x_width=64):
    samples = samples.reshape(nx, ny, x_height, time_steps, x_width, -1)
    samples = np.concatenate(samples, 1)
    samples = np.concatenate(samples, 2)
    samples = np.transpose(samples, [1, 0, 2, 3])[..., :3]

    fig, ax = plt.subplots(figsize=(ny, nx))
    im = ax.imshow(samples[0])
    ax.set_axis_off()
    fig.tight_layout()

    def init():
        im.set_data(samples[0])
        return (im,)

    def animate(i):
        im.set_data(samples[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=time_steps, interval=100,
                                   blit=True)
    plt.close()
    return HTML(anim.to_html5_video())


def samples_to_video_with_cap(samples, nx, ny, time_steps=16, input_t=6, x_height=64, x_width=64):
    samples = samples.reshape(nx, ny, x_height, time_steps, x_width, -1)
    samples = np.concatenate(samples, 1)
    samples = np.concatenate(samples, 2)
    samples = np.transpose(samples, [1, 0, 2, 3])[..., :3]

    fig, ax = plt.subplots(figsize=(ny, nx))
    ax.set_axis_off()
    fig.tight_layout()

    ims = []

    for i in range(time_steps):
        frame = ax.imshow(samples[i])
        if i < input_t:
            t = ax.annotate("Truth t={}".format(i + 1), (5, 2.5), bbox=dict(boxstyle="round", fc="w"))  # add text
        else:
            t = ax.annotate("Prediction t+{}".format(i - input_t + 1), (5, 2.5), bbox=dict(boxstyle="round", fc="w"))
        ims.append([frame, t])
    anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True, repeat_delay=100)
    plt.close()
    return HTML(anim.to_html5_video())
