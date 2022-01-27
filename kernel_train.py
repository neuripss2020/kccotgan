#!/usr/bin/env python

import argparse
import data_utils
import gan
import gan_utils
import glob
import os
import time
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

print('Tensorflow version:', tf.__version__)
tf.keras.backend.set_floatx('float32')

start_time = time.time()


def train(args):
    test = args.test
    dname = args.dname
    batch_size = args.batch_size
    seed = args.seed
    save_freq = args.save_freq

    # filter size for (de)convolutional layers
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    reg_penalty = args.reg_penalty
    g_output_activation = 'sigmoid'
    nlstm = args.n_lstm
    x_height = args.height
    x_width = args.width
    channels = args.n_channels
    epochs = args.n_epochs
    bn = args.batch_norm

    dataset = dname + '-cot'
    # Number of RNN layers stacked together
    n_layers = 1
    gen_lr = args.lr
    disc_lr = args.lr
    tf.random.set_seed(seed)
    np.random.seed(seed)

    it_counts = 0
    warmup_step = args.warmup
    decay_steps = 5000
    decay_rate = 0.975
    # decaying learning rate scheme
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=gen_lr, decay_steps=decay_steps,
                                                                 decay_rate=decay_rate, staircase=True)
    warmup_schedule = data_utils.WarmUp(initial_learning_rate=gen_lr, decay_schedule_fn=lr_schedule, warmup_steps=warmup_step)

    # Add gradient clipping before updates
    gen_optimiser = tf.keras.optimizers.Adam(warmup_schedule, beta_1=0.5, beta_2=0.9)
    dischm_optimiser = tf.keras.optimizers.Adam(warmup_schedule, beta_1=0.5, beta_2=0.9)

    disc_iters = 1
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    total_time_steps = args.total_time_steps
    int_time_steps = args.int_time_steps
    pred_time_steps = total_time_steps - int_time_steps
    scaling_coef = 1.0 / args.scaling_coef
    # dropout rates
    dp = args.dropout
    rnn_dp = args.rnn_dropout
    regularization = args.regularization
    cw = args.clockwork
    kernel_choice = args.kernel
    init_sig = args.init_sigma
    z_channels = args.z_channels

    # adjust channel parameter as we want to drop the
    # alpha channel for animated Sprites
    batched_x = None
    if dname == 'penn_action':
        dataset = tf.data.Dataset.from_generator(data_utils.load_penn_data,
                                                 args=([batch_size, x_height, x_width, total_time_steps]),
                                                 output_types=tf.float64)
        batched_x = dataset.batch(batch_size).repeat(epochs)
    elif dname == 'kth':
        dataset = tf.data.Dataset.from_generator(data_utils.load_kth_data,
                                                 args=([batch_size, x_height, x_width, total_time_steps, True]),
                                                 output_types=tf.float64)
        batched_x = dataset.batch(batch_size).repeat(epochs)

        dataset = tf.data.Dataset.from_generator(data_utils.load_kth_data,
                                                 args=([batch_size, x_height, x_width, total_time_steps, False]),
                                                 output_types=tf.float64)
        test_x = dataset.batch(batch_size).repeat(epochs)
    elif dname == "mmnist":
        data_path = "../data/mmnist/mnist_training_set.npy"
        training_data = np.load(data_path) / 255.0
        training_data = tf.transpose(training_data[:total_time_steps, ...], (1, 0, 2, 3))
        training_data = tf.transpose(training_data, (0, 2, 1, 3))
        dataset = tf.data.Dataset.from_tensor_slices(training_data)
        batched_x = dataset.batch(batch_size).repeat(epochs)

        data_path = "../data/mmnist/mnist_test_set.npy"
        test_data = np.load(data_path) / 255.0
        test_data = tf.transpose(test_data[:total_time_steps, ...], (1, 0, 2, 3))
        test_data = tf.transpose(test_data, (0, 2, 1, 3))
        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_x = dataset.batch(batch_size).repeat(epochs)
    elif dname == "mazes":
        # path to data
        root_path = '../data/'
        data_reader = data_utils.DataReader(dataset=dname, time_steps=total_time_steps,
                                            root=root_path, custom_frame_size=x_height, mode="train")
        batched_x = data_reader.provide_dataset(batch_size=batch_size)

        data_path = "../data/mazes/np_mazes_test.npy"
        test_data = np.load(data_path)[:, :, :total_time_steps, :, :]
        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_x = dataset.batch(batch_size).repeat(epochs)
    elif dname == "robot_push":
        # data = data_utils.robot_push_data()
        dataset = tf.data.Dataset.from_generator(data_utils.robot_push_data, args=([total_time_steps, True]),
                                                 output_types=tf.float64)
        batched_x = dataset.batch(batch_size).repeat(epochs)
        test = tf.data.Dataset.from_generator(data_utils.robot_push_data, args=([total_time_steps, False]),
                                              output_types=tf.float64)
        test_x = test.batch(batch_size).repeat(epochs)

    encode_period = [int(x) for x in args.enc_period.split(",")]
    decode_period = [int(x) for x in args.dec_period.split(",")]
    z_height = 4
    z_width = 4

    # Define a standard multivariate normal for (z1, z2, ..., zT) --> (y1, y2, ..., yT)
    dist_z = tfp.distributions.Normal(0.0, 1.0)

    context_encoder = gan.VideoEncoderConvLSTM(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width,
                                               x_height, z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm,
                                               nchannel=channels, dropout=dp, rnn_dropout=rnn_dp, reg=regularization,
                                               cw=cw, period=encode_period)

    decoder = gan.VideoDecoderConvLSTM(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width, x_height,
                                       z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels,
                                       dropout=dp, rnn_dropout=rnn_dp, output_activation=g_output_activation,
                                       reg=regularization, cw=cw, period=decode_period)

    discriminator_h = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                             z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)
    discriminator_m = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                             z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)

    if args.checkpoint:
        enc_ckpt_path = 'trained/cot/{}_encoder/'.format(args.ckpt_name)
        encoder.load_weights(enc_ckpt_path)
        dec_ckpt_path = 'trained/cot/{}_decoder/'.format(args.ckpt_name)
        decoder.load_weights(dec_ckpt_path)
        h_ckpt_path = 'trained/cot/{}/'.format(args.ckpt_name + '_h')
        discriminator_h.load_weights(h_ckpt_path)
        m_ckpt_path = 'trained/cot/{}/'.format(args.ckpt_name + '_m')
        discriminator_m.load_weights(m_ckpt_path)
        print('Checkpoints loaded. Training resumed.')
    else:
        print('New training started.')

    # data_utils.check_model_summary(batch_size, z_dims, generator)
    # data_utils.check_model_summary(batch_size, seq_len, discriminator_h)

    f_name = "{}_lr{}_lam{}_{}kernel_init_sig{}_{}".format(dname, gen_lr, reg_penalty, kernel_choice, init_sig,
                                                           args.ckpt_str)

    saved_file = f_name + "-{}{}-{}:{}:{}.{}".format(datetime.now().strftime("%h"), datetime.now().strftime("%d"),
                                                     datetime.now().strftime("%H"), datetime.now().strftime("%M"),
                                                     datetime.now().strftime("%S"), datetime.now().strftime("%f"))

    if args.mixed_sinkhorn:
        model_fn = f_name + "-mixed"
    elif args.bi_causal:
        model_fn = f_name + "-bicausal"
    else:
        model_fn = f_name + "-no_mix"

    log_dir = "trained/{}/log".format(saved_file)

    # Create directories for storing images later.
    if not os.path.exists("trained/{}/data".format(saved_file)):
        os.makedirs("trained/{}/data".format(saved_file))
    if not os.path.exists("trained/{}/images".format(saved_file)):
        os.makedirs("trained/{}/images".format(saved_file))

    # GAN train notes
    with open("./trained/{}/train_notes.txt".format(saved_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
            dataset,
            total_time_steps, ))
        f.write("STATE_SIZE: {}\nNUM_LAYERS: {}\nLAMBDA: {}\n".format(
            g_state_size,
            n_layers,
            reg_penalty))
        f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\nDiscriminator LR:{}\n".format(
            batch_size,
            disc_iters,
            gen_lr,
            disc_lr))
        f.write("SINKHORN EPS: {}\nSINKHORN L: {}\n\n".format(
            sinkhorn_eps,
            sinkhorn_l))

    train_writer = tf.summary.create_file_writer(logdir=log_dir)

    gaussian_kernel = data_utils.KernelSmoothing(temporal_kernel_size=6, spatial_kernel_size=6)

    # @tf.function
    def disc_training_step(real_in, real_pred, sigma):
        hidden_z = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height, z_width, z_channels])
        with tf.GradientTape() as disc_tape:
            real_inp = tf.concat((real_in, real_pred), axis=2)
            context_features, preds_features = context_encoder.call(real_inp)
            fake_pred = decoder.call(context_features, preds_features, hidden_z)

            real = tf.concat((real_in, real_pred), axis=2)
            fake = tf.concat((real_in, fake_pred), axis=2)

            if kernel_choice == '1d':
                real = gaussian_kernel.temporal_convolution(real, sigma)
                fake = gaussian_kernel.temporal_convolution(fake, sigma)

            elif kernel_choice == '2d':
                real = gaussian_kernel.spatial_convolution(real, sigma)
                fake = gaussian_kernel.spatial_convolution(fake, sigma)

            elif kernel_choice == '3d':
                real = gaussian_kernel.gaussian_convolution3D(real, sigma)
                fake = gaussian_kernel.gaussian_convolution3D(fake, sigma)

            h_fake = discriminator_h.call(fake)
            h_real = discriminator_h.call(real)

            m_real = discriminator_m.call(real)
            m_fake = discriminator_m.call(fake)

            loss = gan_utils.compute_sinkhorn_loss(real, fake, scaling_coef, sinkhorn_eps, sinkhorn_l, h_fake, m_real,
                                                   h_real, m_fake, video=True)
            pm1 = gan_utils.scale_invariante_martingale_regularization(m_real, reg_penalty, scaling_coef)
            disc_loss = - loss + pm1
            # update discriminator parameters
        disch_grads, discm_grads = disc_tape.gradient(disc_loss, [discriminator_h.trainable_variables,
                                                                  discriminator_m.trainable_variables])
        dischm_optimiser.apply_gradients(zip(disch_grads, discriminator_h.trainable_variables))
        dischm_optimiser.apply_gradients(zip(discm_grads, discriminator_m.trainable_variables))
        return pm1

    # @tf.function
    def gen_training_step(real_in, real_pred, sigma):
        hidden_z = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height, z_width, z_channels])

        with tf.GradientTape() as gen_tape:
            real_inp = tf.concat((real_in, real_pred), axis=2)
            context_features, preds_features = context_encoder.call(real_inp)
            fake_pred = decoder.call(context_features, preds_features, hidden_z)

            real = tf.concat((real_in, real_pred), axis=2)
            fake = tf.concat((real_in, fake_pred), axis=2)
            if kernel_choice == '1d':
                real = gaussian_kernel.temporal_convolution(real, sigma)
                fake = gaussian_kernel.temporal_convolution(fake, sigma)

            elif kernel_choice == '2d':
                real = gaussian_kernel.spatial_convolution(real, sigma)
                fake = gaussian_kernel.spatial_convolution(fake, sigma)

            elif kernel_choice == '3d':
                real = gaussian_kernel.gaussian_convolution3D(real, sigma)
                fake = gaussian_kernel.gaussian_convolution3D(fake, sigma)

            h_fake = discriminator_h.call(fake)
            h_real = discriminator_h.call(real)

            m_real = discriminator_m.call(real)
            m_fake = discriminator_m.call(fake)

            loss = gan_utils.compute_sinkhorn_loss(real, fake, scaling_coef, sinkhorn_eps, sinkhorn_l, h_fake, m_real,
                                                   h_real, m_fake, video=True)
        con_grads, dec_grads = gen_tape.gradient(loss, [context_encoder.trainable_variables, decoder.trainable_variables])
        gen_optimiser.apply_gradients(zip(con_grads, context_encoder.trainable_variables))
        gen_optimiser.apply_gradients(zip(dec_grads, decoder.trainable_variables))
        return loss

    with tqdm.trange(epochs, ncols=100, unit="epoch") as ep:
        for _ in ep:
            it = tqdm.tqdm(ncols=100)
            for x in batched_x:
                if x.shape[0] != batch_size:
                    continue
                it_counts += 1
                real_data = tf.reshape(x, [batch_size, x_height, total_time_steps, x_width, -1])
                # throw away alpha channel
                real_data = tf.cast(real_data[..., :channels], tf.float32)
                # split real data to training inputs and predictions
                real_inputs = real_data[:, :, :int_time_steps, :, :]
                real_preds = real_data[:, :, int_time_steps:, :, :]

                if args.decaying_sigma:
                    sig = gaussian_kernel.annealing_sigma(init_sig, it_counts)
                else:
                    sig = init_sig

                pm = disc_training_step(real_inputs, real_preds, sig)
                loss = gen_training_step(real_inputs, real_preds, sig)
                it.set_postfix(loss=float(loss))
                it.update(1)

                with train_writer.as_default():
                    tf.summary.scalar('pM', pm, step=it_counts)
                    tf.summary.scalar('Sinkhorn Loss', loss, step=it_counts)
                    train_writer.flush()

                if not np.isfinite(loss.numpy()):
                    print('%s Loss exploded!' % model_fn)
                    # Open the existing file with mode a - append
                    with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                        # Include any experiment notes here:
                        f.write("\n Training failed! ")
                    break
                else:
                    if it_counts % save_freq == 0 or it_counts == 1:
                        if it_counts % 10000 == 0 and it_counts > 9999:
                            context_encoder.save_weights("./trained/{}/{}_con_encoder_{}iters/".format(test, model_fn,
                                                                                                       it_counts))
                            decoder.save_weights("./trained/{}/{}_iter{}_decoder_{}iters/".format(test, model_fn,
                                                                                                  it_counts, it_counts))
                            discriminator_h.save_weights("./trained/{}/{}_iter{}_h/".format(test, model_fn, it_counts))
                            discriminator_m.save_weights("./trained/{}/{}_iter{}_m/".format(test, model_fn, it_counts))
                        for x in test_x.take(1):
                            test_data = tf.reshape(x, [batch_size, x_height, total_time_steps, x_width, -1])
                            # throw away alpha channel
                            test_data = tf.cast(test_data[..., :channels], tf.float32)
                            # split real data to training inputs and predictions
                            test_inputs = test_data[:, :, :int_time_steps, :, :]

                            for i in range(pred_time_steps):
                                context_features, preds_features = context_encoder.call(test_inputs, training=False)
                                hidden_z = dist_z.sample([batch_size, 1, 4, 4, 128])
                                preds = decoder.call(context_features, preds_features, hidden_z, training=False)
                                test_inputs = tf.concat((test_inputs, preds), axis=2)

                        images = tf.reshape(test_inputs, [batch_size, x_height, x_width * total_time_steps, channels])
                        # plot first 10 samples within one image
                        img = tf.concat(list(images[:min(10, batch_size)]), axis=0)[None]
                        with train_writer.as_default():
                            tf.summary.image("Training data", img, step=it_counts)

            print("--- The entire training takes %s minutes ---" % ((time.time() - start_time) / 60.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cot')

    parser.add_argument('-d', '--dname', type=str, default='robot_push',
                        choices=['animation', 'human_action', 'ucf', 'kth', 'penn_action', 'mmnist', 'mazes', 'robot_push'])
    parser.add_argument('-t', '--test',  type=str, default='cot', choices=['cot'])
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-gss', '--g_state_size', type=int, default=8)
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=8)
    parser.add_argument('-dss', '--d_state_size', type=int, default=8)
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=8)
    # animation data has T=13 and human action data has T=16
    parser.add_argument('-tts', '--total_time_steps', type=int, default=15)
    parser.add_argument('-its', '--int_time_steps', type=int, default=5)
    parser.add_argument('-gts', '--gen_time_steps', type=int, default=10)
    parser.add_argument('-nch', '--n_channels', type=int, default=3)
    parser.add_argument('-nz', '--z_channels', type=int, default=128)
    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
    parser.add_argument('-reg_p', '--reg_penalty', type=float, default=1.0)
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default='../data/animation/*.tfrecord')
    parser.add_argument('-save', '--save_freq', type=int, default=10)
    parser.add_argument('-lr', '--lr', type=float, default=5e-4)
    parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
    parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
    parser.add_argument('-dp', '--dropout', type=float, default=0.0)
    parser.add_argument('-rdp', '--rnn_dropout', type=float, default=0.0)
    parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
    # Scale parameter applied will be 1.0 / scaling_coef
    parser.add_argument('-sc', '--scaling_coef', type=float, default=15.0)
    parser.add_argument('-mix', '--mixed_sinkhorn', type=bool, default=False)
    parser.add_argument('-ckpt', '--checkpoint', type=bool, default=False)
    parser.add_argument('-cn', '--ckpt_name', type=str, default='ckpts name')
    parser.add_argument('-bc', '--bi_causal', type=bool, default=False)
    parser.add_argument('-k', '--kernel', type=str, default="none", choices=['1d', '2d', '3d', 'none'])
    parser.add_argument('-cw', '--clockwork', type=bool, default=False)
    parser.add_argument('-regu', '--regularization', type=bool, default=False)
    parser.add_argument('-ct', '--ckpt_str', type=str, default='full_training')
    parser.add_argument('-xh', '--height', type=int, default=64)
    parser.add_argument('-xw', '--width', type=int, default=64)
    parser.add_argument('-ne', '--n_epochs', type=int, default=100)
    parser.add_argument('-wu', '--warmup', type=int, default=10000)
    parser.add_argument('-epd', '--enc_period', type=str, default="1,1,1,1")
    parser.add_argument('-dpd', '--dec_period', type=str, default="1,1,1,1")
    parser.add_argument('-nstd', '--n_std', type=float, default=0.1)
    parser.add_argument('-isig', '--init_sigma', type=float, default=5.0)
    parser.add_argument('-desig', '--decaying_sigma', type=bool, default=False)

    args = parser.parse_args()

    train(args)
