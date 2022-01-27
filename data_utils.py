#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from absl import logging

from scipy.stats import multivariate_normal
from functools import partial
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import rbf_kernel
import random
import collections
import os
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import animation, rc
from IPython.display import HTML
from scipy import signal
from typing import Callable
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


def robot_push_data(T=30, train=True):
    FRAMES_PER_VIDEO = 30
    IMG_SHAPE = (64, 64, 3)
    if train:
        filedir = '../data/softmotion30_44k/train/'
    else:
        filedir = '../data/softmotion30_44k/test/'
    logging.info("Reading data from %s.", filedir)
    files = tf.io.gfile.listdir(filedir)
    logging.info("%d files found.", len(files))

    # For each file
    for filename in sorted(tf.io.gfile.listdir(filedir)):
        filepath = os.path.join(filedir, filename)

        # For each video inside the file
        for video_id, example_str in enumerate(
                tf.compat.v1.io.tf_record_iterator(filepath)):
            example = tf.train.SequenceExample.FromString(example_str)

            # Merge all frames together
            all_frames = []
            for frame_id in range(FRAMES_PER_VIDEO):
                # Extract all features from the original proto context field
                frame_feature = {  # pylint: disable=g-complex-comprehension
                    out_key: example.context.feature[in_key.format(frame_id)]
                    # pylint: disable=g-complex-comprehension
                    for out_key, in_key in [
                        ("image_main", "{}/image_main/encoded"),
                        ("image_aux1", "{}/image_aux1/encoded"),
                        ("endeffector_pos", "{}/endeffector_pos"),
                        ("action", "{}/action"),
                    ]
                }

                # Decode float
                for key in ("endeffector_pos", "action"):
                    values = frame_feature[key].float_list.value
                    frame_feature[key] = [values[i] for i in range(len(values))]

                # Decode images (from encoded string)
                for key in ("image_main", "image_aux1"):
                    img = frame_feature[key].bytes_list.value[0]  # pytype: disable=attribute-error
                    img = np.frombuffer(img, dtype=np.uint8)
                    img = np.reshape(img, IMG_SHAPE)
                    frame_feature[key] = img
                all_frames.append(frame_feature["image_aux1"])
            all_frames = np.stack(all_frames).transpose(1, 0, 2, 3) / 255.0
            yield all_frames[:, :T, :, :]


def load_penn_data(batch_size=2, height=128, width=128, time_step=30, crop=True):
    data_path = '../data/penn_frames'
    list_f = [x[0] for x in os.walk(data_path)]
    folders = list_f[1:]
    for x in range(batch_size):
        frames = []
        rand_folder = random.choice(folders)
        files_in_folder = [x[0] for x in os.walk(data_path + '/' + rand_folder)][1:]
        if (len(files_in_folder) // 2) < time_step:
            for i in range(1, time_step + 1):
                idx = str(i)
                if len(idx) == 1:
                    file_name = '00000' + idx + '.jpg'
                elif len(idx) == 2:
                    file_name = '0000' + idx + '.jpg'
                else:
                    continue
                if len(frames) > time_step:
                    break
                path_to_jpg = rand_folder + '/' + file_name
                img = plt.imread(path_to_jpg)
                frame = img / 255.0
                if crop:
                    frame = tf.image.resize_with_crop_or_pad(frame, height, width)
                frames.append(frame)
        else:
            for i in range(1, time_step * 2 + 1, 2):
                idx = str(i)
                if len(idx) == 1:
                    file_name = '00000' + idx + '.jpg'
                elif len(idx) == 2:
                    file_name = '0000' + idx + '.jpg'
                else:
                    continue
                if len(frames) > time_step:
                    break
                path_to_jpg = rand_folder + '/' + file_name
                img = plt.imread(path_to_jpg)
                frame = img / 255.0
                if crop:
                    frame = tf.image.resize_with_crop_or_pad(frame, height, width)
                frames.append(frame)
        cropped_frames = np.reshape(frames, newshape=(time_step, height, width, 3))
        cropped_frames = np.transpose(cropped_frames, (1, 0, 2, 3))
        cropped_frames = np.reshape(cropped_frames, newshape=(height, time_step * width, 3))
        yield cropped_frames


def load_kth_data(batch_size, height=64, width=64, time_step=16, train=True):
    if train:
        path = '../data/kth'
    else:
        path = '../data/kth_test'
    list_f = [x for x in os.listdir(path)]

    for x in range(batch_size):
        rand_folder = random.choice(list_f)
        path_to_file = path + '/' + rand_folder
        file_name = random.choice(os.listdir(path_to_file))
        path_to_video = path_to_file + '/' + file_name
        vidcap = cv2.VideoCapture(path_to_video)
        n_frames = vidcap.get(7)
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


def save_low_d(data, saved_file, input_len=25, row=4, col=4, real=False):
    bs, ts, ds = data.shape
    x = np.arange(ts)
    n = 0

    colors = []
    for i in range(ts):
        if i < (input_len-1):
            colors.append('c')
        else:
            colors.append('r')
    cmap = ListedColormap(colors)

    fig, axs = plt.subplots(row, col, figsize=(12, 6))

    for r in range(row):
        for c in range(col):
            points = np.array([x, np.squeeze(data[n, ...])]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, linewidth=2)
            # set color to date values
            lc.set_array(x)
            axs[r, c].add_collection(lc)
            axs[r, c].autoscale_view()
            n += 1
            if r == (row - 1):
                axs[r, c].set_xticks([0, input_len-1, ts-1])
                axs[r, c].set_xticklabels(["1", "{}".format(input_len), "{}".format(ts)])
                axs[r, c].set_xlabel("t")
                axs[r, c].tick_params(top='off', bottom='on', left='off', right='off', labelleft='off',
                                      labelbottom='on')
                axs[r, c].set(frame_on=False)
            else:
                axs[r, c].axis('off')
    if real:
        plt.savefig("./trained/{}/images/real.png".format(saved_file))
    else:
        plt.savefig("./trained/{}/images/preds.png".format(saved_file))
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


"""Minimal data reader for GQN TFRecord datasets. 
   Adapted from the original code: https://github.com/deepmind/gqn-datasets/blob/master/data_reader.py."""
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
    return HTML(anim.to_jshtml())


class KernelSmoothing:
    def __init__(self, temporal_kernel_size=6, spatial_kernel_size=8):
        self.temporal_radius = temporal_kernel_size // 2
        self.spatial_radius = spatial_kernel_size // 2

    def gaussian_kernel1d(self, radius, sigma):
        """
        Computes a 1-D Gaussian convolution kernel.
        """
        sigma2 = sigma * sigma
        x = tf.range(-radius, radius+1, dtype=tf.float32)
        kernel = tf.exp(-0.5 / sigma2 * x ** 2)
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel

    def gaussian_kernel3d(self, radius, sigma):
        sigma2 = sigma * sigma
        x = tf.range(-radius, radius+1, dtype=tf.float32)
        y = tf.range(-radius, radius+1, dtype=tf.float32)
        z = tf.range(-radius, radius+1, dtype=tf.float32)
        xx, yy, zz = tf.meshgrid(x, y, z)
        kernel = tf.exp(- 0.5 / sigma2 * (xx ** 2 + yy ** 2 + zz ** 2))[:, :, :, tf.newaxis, tf.newaxis]
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel

    def temporal_convolution(self, inputs, sigma):
        weights = self.gaussian_kernel1d(self.temporal_radius, sigma)[:, tf.newaxis, tf.newaxis]
        # batch_shape + [in_width, in_channels]
        bs, h, t, w, nc = inputs.shape
        inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
        inputs = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])

        inputs = tf.reshape(inputs, [bs * h * w * nc, t, 1])

        paddings = tf.constant([[0, 0], [self.temporal_radius, self.temporal_radius], [0, 0]])
        inputs = tf.pad(inputs, paddings, "REFLECT")

        smoothed = tf.nn.conv1d(inputs, weights, stride=1, padding='VALID')

        smoothed = tf.reshape(smoothed, [bs, h, w, nc, t])

        smoothed = tf.transpose(smoothed, perm=[0, 1, 2, 4, 3])
        smoothed = tf.transpose(smoothed, perm=[0, 1, 3, 2, 4]) / tf.reduce_max(smoothed)
        return smoothed

    def spatial_convolution(self, inputs, sigma):
        weights = self.gaussian_kernel1d(self.spatial_radius, sigma)
        kernel = np.tensordot(weights, weights, 0)[:, :, tf.newaxis, tf.newaxis]

        # batch_shape + [in_height, in_width, in_channels] and a filter
        # kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
        bs, h, t, w, nc = inputs.shape
        if nc > 1:
            inputs = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
            inputs = tf.reshape(inputs, [bs * nc * t, h, w, 1])

            smoothed = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
            smoothed = tf.reshape(smoothed, [bs, nc, t, h, w])

            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4])
            smoothed = tf.transpose(smoothed, perm=[0, 1, 3, 2, 4])
            smoothed = tf.transpose(smoothed, perm=[0, 1, 2, 4, 3])
            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4]) / tf.reduce_max(smoothed)
        else:
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
            inputs = tf.reshape(inputs, [bs * t, h, w, 1])
            smoothed = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
            smoothed = tf.reshape(smoothed, [bs, t, h, w, nc])
            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4]) / tf.reduce_max(smoothed)
        return smoothed

    def gaussian_convolution3D(self, inputs, sigma):
        kernel = self.gaussian_kernel3d(self.spatial_radius, sigma)
        bs, h, t, w, nc = inputs.shape
        if nc > 1:
            inputs = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
            inputs = tf.reshape(inputs, [bs * nc, t, h, w, 1])

            paddings = tf.constant([[0, 0], [self.spatial_radius, self.spatial_radius],
                                    [self.spatial_radius, self.spatial_radius],
                                    [self.spatial_radius, self.spatial_radius], [0, 0]])
            inputs = tf.pad(inputs, paddings, "REFLECT")

            smoothed = tf.nn.conv3d(inputs, kernel, [1, 1, 1, 1, 1], padding='VALID')
            smoothed = tf.reshape(smoothed, [bs, nc, t, h, w])

            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4])
            smoothed = tf.transpose(smoothed, perm=[0, 1, 3, 2, 4])
            smoothed = tf.transpose(smoothed, perm=[0, 1, 2, 4, 3])
            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4]) / tf.reduce_max(smoothed)
        else:
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
            paddings = tf.constant([[0, 0], [self.spatial_radius, self.spatial_radius],
                                    [self.spatial_radius, self.spatial_radius],
                                    [self.spatial_radius, self.spatial_radius], [0, 0]])
            inputs = tf.pad(inputs, paddings, "REFLECT")
            smoothed = tf.nn.conv3d(inputs, kernel, [1, 1, 1, 1, 1], padding='VALID')
            smoothed = tf.transpose(smoothed, perm=[0, 2, 1, 3, 4]) / tf.reduce_max(smoothed)
        return smoothed

    def annealing_sigma(self, init_sigma, step, decay_steps=500, decay_rate=0.975):
        decaying_sigma = init_sigma * decay_rate ** (step / decay_steps)
        return decaying_sigma


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate: float, decay_schedule_fn: Callable, warmup_steps: int, power: float = 1.0,
                 name: str = None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def exponential_decay_with_warmup(warmup_step, learning_rate_base, global_step, learning_rate_step,
                                  learning_rate_decay, staircase=False):
    linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)
    # warm up step is the iteration where we start decaying lr
    exponential_decay = tf.compat.v1.train.exponential_decay(learning_rate_base, global_step - warmup_step,
                                                             learning_rate_step, learning_rate_decay,
                                                             staircase=staircase)
    learning_rate = tf.cond(global_step <= warmup_step, lambda: linear_increase,
                            lambda: exponential_decay)
    return learning_rate










