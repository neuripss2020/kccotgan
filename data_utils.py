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


def sine_data_generation(n_samples, seq_len, dims):
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(n_samples):
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(dims):
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0.1, 1.0)
            phase1 = np.random.uniform(-math.pi, math.pi)

            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(seq_len)]
            Temp.append(Temp1)

        # Align row/column
        Temp = np.transpose(np.asarray(Temp))

        # Normalize to [0,1]
        Temp = (Temp + 1) * 0.5

        data.append(Temp)
    return np.array(data)


def changepoint_pdf(Y, cov_ms, cov_Ms):
    """
    """
    seq_length = Y.shape[0]
    logpdf = []
    for (i, m) in enumerate(range(int(seq_length/2), seq_length-1)):
        Y_m = Y[:m, 0]
        Y_M = Y[m:, 0]
        M = seq_length - m
        # generate mean function for second part
        Ymin = np.min(Y_m)
        initial_val = Y_m[-1]
        if Ymin > 1:
            final_val = (1.0 - M/seq_length)*Ymin
        else:
            final_val = (1.0 + M/seq_length)*Ymin
        mu_M = np.linspace(initial_val, final_val, M)
        # ah yeah
        logpY_m = multivariate_normal.logpdf(Y_m, mean=np.zeros(m), cov=cov_ms[i])
        logpY_M = multivariate_normal.logpdf(Y_M, mean=mu_M, cov=cov_Ms[i])
        logpdf_m = logpY_m + logpY_M
        logpdf.append(logpdf_m)
    return logpdf


def changepoint(seq_length=30, num_samples=28*5*100):
    """
    Generate data from two GPs, roughly speaking.
    The first part (up to m) is as a normal GP.
    The second part (m to end) has a linear downwards trend conditioned on the
    first part.
    """
    print('Generating samples from changepoint...')
    T = np.arange(seq_length)
    # sample breakpoint from latter half of sequence
    m_s = np.random.choice(np.arange(int(seq_length/2), seq_length-1), size=num_samples)
    samples = np.zeros(shape=(num_samples, seq_length, 1))
    # kernel parameters and stuff
    gamma=5.0/seq_length
    A = 0.01
    sigmasq = 0.8*A
    lamb = 0.0  # if non-zero, cov_M risks not being positive semidefinite...
    kernel = partial(rbf_kernel, gamma=gamma)
    # multiple values per m
    N_ms = []
    cov_ms = []
    cov_Ms = []
    pdfs = []
    for m in range(int(seq_length/2), seq_length-1):
        # first part
        M = seq_length - m
        T_m = T[:m].reshape(m, 1)
        cov_m = A*kernel(T_m.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_ms.append(cov_m)
        # the second part
        T_M = T[m:].reshape(M, 1)
        cov_mM = kernel(T_M.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_M = sigmasq*(np.eye(M) - lamb*np.dot(np.dot(cov_mM, np.linalg.inv(cov_m)), cov_mM.T))
        cov_Ms.append(cov_M)
    for n in range(num_samples):
        m = m_s[n]
        M = seq_length-m
        # sample the first m
        cov_m = cov_ms[m - int(seq_length/2)]
        Xm = multivariate_normal.rvs(cov=cov_m)
        # generate mean function for second
        Xmin = np.min(Xm)
        initial_val = Xm[-1]
        if Xmin > 1:
            final_val = (1.0 - M/seq_length)*Xmin
        else:
            final_val = (1.0 + M/seq_length)*Xmin
        mu_M = np.linspace(initial_val, final_val, M)
        # sample the rest
        cov_M = cov_Ms[m -int(seq_length/2)]
        XM = multivariate_normal.rvs(mean=mu_M, cov=cov_M)
        # combine the sequence
        # NOTE: just one dimension
        samples[n, :, 0] = np.concatenate([Xm, XM])
    pdf = partial(changepoint_pdf, cov_ms=cov_ms, cov_Ms=cov_Ms)
    return samples, pdf, m_s


def gaussian_process(seq_length=30, num_samples=28*5*100, num_signals=1, scale=0.2, kernel='rbf', **kwargs):
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.empty(shape=(num_samples, seq_length, num_signals))
    # T = np.arange(seq_length)/seq_length    # note, between 0 and 1
    T = np.arange(seq_length)    # note, not between 0 and 1
    if kernel == 'periodic':
        cov = periodic_kernel(T)
    elif kernel =='rbf':
        cov = rbf_kernel(T.reshape(-1, 1), gamma=scale)
    else:
        raise NotImplementedError
    # scale the covariance
    cov *= 0.1
    # define the distribution
    mu = np.zeros(seq_length)
    # print(np.linalg.det(cov))
    distribution = multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
    pdf = distribution.logpdf
    # now generate samples
    for i in range(num_signals):
        samples[:, :, i] = distribution.rvs(size=num_samples)
    return samples, pdf


class Brownian:
    """
    A Brownian motion class constructor
    """

    def __init__(self, x0=0):
        """
        Init class
        """
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"

        self.x0 = float(x0)

    def gen_random_walk(self, batch_size=16, n_step=100):
        """
        Generate motion by random walk

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones([batch_size, n_step], dtype=np.float32) * self.x0

        for i in range(1, n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1], size=batch_size).astype(np.float32)
            # Weiner process
            w[:, i] = w[:, i - 1] + (yi / np.sqrt(n_step))
        # normalise it
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        return w

    def gen_normal(self, batch_size=16, n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        Arguments:
            n_step: Number of steps
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones([batch_size, n_step], dtype=np.float32) * self.x0

        for i in range(1, n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal(size=batch_size).astype(np.float32)
            # Weiner process
            w[:, i] = w[:, i - 1] + (yi / np.sqrt(n_step))
        # normalise it
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        return w


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


class Gaussian:
    def __init__(self, D=1):
        self.D = D

    def batch(self, batch_size):
        return np.random.randn(batch_size, 1, self.D)


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


class SineImage(object):
    '''
    :param Dx: dimensionality of of data at each time step
    :param angle: rotation
    :param z0: initial position and velocity
    :param rand_std: gaussian randomness in the latent trajectory
    :param noise_std: observation noise at output
    '''
    def __init__(self, Dx=20, angle=np.pi / 6., z0=None, rand_std=0.0, noise_std=0.0, length=None, amp=1.0):
        super().__init__()
        self.D = 2
        self.Dx = Dx
        self.z0 = z0

        self.A = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.rand_std = rand_std
        self.noise_std = noise_std
        self.length = length
        self.amp = amp

    def sample(self, n, T):
        # n: number of samples
        # T: lenght of each sample
        if self.z0 is None:
            z = np.random.randn(n, 2)
            z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        else:
            z = np.tile(self.z0, (n, 1))

        zs = []
        for t in np.arange(T):
            m = self.conditional_param(z)
            z = m + np.random.randn(*m.shape) * self.rand_std
            zs += z,

        zs = np.stack(zs, 1)

        grid = np.linspace(-1.5, 1.5, self.Dx)

        mean = np.exp(- 0.5 * (zs[..., :1] - grid) ** 2 / 0.3 ** 2) * self.amp
        mean = mean.reshape(n, -1)
        xs = mean + np.random.randn(*mean.shape) * self.noise_std

        return zs, xs.reshape(n, T, self.Dx)

    def conditional_param(self, zt):

        slope = 1.0
        r = np.sqrt(np.sum(zt ** 2, -1))
        r_ratio = 1.0 / (np.exp(-slope * 4 * (r - 0.3)) + 1) / r

        ztp1 = np.matmul(zt, self.A)
        ztp1 *= r_ratio[..., None]

        return ztp1

    def batch(self, batch_size):
        return self.sample(batch_size, self.length)[1]


class NPData(object):
    def __init__(self, data, batch_size, nepoch=np.inf, tensor=True):
        self.data = data
        self.N, self.length = data.shape[0:2]
        self.epoch = 0
        self.counter = 0
        np.random.shuffle(self.data)
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.tensor = tensor

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.N / self.batch_size))

    def __next__(self):
        if (self.counter + 1) * self.batch_size > self.N:
            self.epoch += 1
            np.random.shuffle(self.data)
            self.counter = 0

        if np.isfinite(self.nepoch) and self.epoch == self.nepoch:
            raise StopIteration

        idx = slice(self.counter * self.batch_size, (self.counter + 1) * self.batch_size)
        batch = self.data[idx]
        self.counter += 1
        if self.tensor:
            batch = tf.cast(batch, tf.float32)
        return batch

    def batch(self, batch_size):
        return self.__next__()


class EEGData(NPData):
    '''
    :param Dx: dimensionality of of data at each time step
    :param length: sequence length
    :param batch size: batch size
    '''

    def __init__(self, Dx, length, batch_size, nepoch=np.inf, tensor=True, seed=0, prefix="", downsample=1):
        # nsubject x n trials x channel x times_steps
        all_data = np.load(prefix + "data/eeg_data.npy", allow_pickle=True)
        train_data = []
        test_data = []
        sep = 0.75
        np.random.RandomState(seed).shuffle(all_data)
        for sub_data in all_data:
            ntrial = int(sep * len(sub_data))
            train_data += sub_data[:ntrial, :downsample * length:downsample, :Dx],
            test_data += sub_data[ntrial:, :downsample * length:downsample, :Dx],
            assert train_data[-1].shape[1] == length
            assert train_data[-1].shape[2] == Dx

        self.train_data = self.normalize(train_data)
        self.test_data = self.normalize(test_data)
        self.all_data = np.concatenate([self.train_data, self.test_data], 0)
        super().__init__(self.train_data, batch_size, nepoch, tensor)

    def normalize(self, data):
        data = np.concatenate(data, 0)
        m, s = data.mean((0, 1)), data.std((0, 1))
        data = (data - m) / (3 * s)
        data = np.tanh(data)
        return data


def plot_batch(batch_series, saved_file, axis=None):
    '''
    :param batch_series: a batch of sequence
    :return: plots up to six sequences on shared axis
    '''
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    batch_size = np.shape(batch_series)[0]
    num_seq = np.minimum(len(flatui), batch_size)

    for i in range(0, num_seq):
        data = [_ for _ in enumerate(batch_series[i])]
        sns.lineplot(x=[el[0] for el in data],
                     y=[el[1] for el in data],
                     color=flatui[i % len(flatui)],
                     ax=axis)
    plt.savefig("./trained/{}/images/plot.png".format(saved_file))
    plt.close()


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