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
    path = args.path
    # print(path)
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
    buffer = 200
    bn = args.batch_norm

    dataset = dname + '-cot'
    # Number of RNN layers stacked together
    n_layers = 1
    gen_lr = args.lr
    disc_lr = args.lr
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # decaying learning rate scheme
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=gen_lr, decay_steps=10000,
                                                                 decay_rate=0.985, staircase=True)
    # Add gradient clipping before updates
    gen_optimiser = tf.keras.optimizers.Adam(lr_schedule)
    dischm_optimiser = tf.keras.optimizers.Adam(lr_schedule)
    # gen_optimiser = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5, beta_2=0.9)
    # dischm_optimiser = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5, beta_2=0.9)

    it_counts = 0
    disc_iters = 1
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    total_time_steps = args.total_time_steps
    int_time_steps = args.int_time_steps
    pred_time_steps = total_time_steps - int_time_steps
    generation_steps = args.gen_time_steps
    scaling_coef = 1.0 / args.scaling_coef
    # dropout rates
    dp = args.dropout
    rnn_dp = args.rnn_dropout
    regularization = args.regularization
    cw = args.clockwork

    # adjust channel parameter as we want to drop the
    # alpha channel for animated Sprites
    batched_x = None
    if dname == 'penn_action':
        dataset = tf.data.Dataset.from_generator(data_utils.load_penn_data,
                                                 args=([batch_size, x_height, x_width, total_time_steps]),
                                                 output_types=tf.float64)
        batched_x = dataset.batch(batch_size * 2).repeat(epochs)
    elif dname == 'kth':
        dataset = tf.data.Dataset.from_generator(data_utils.load_kth_data,
                                                 args=([batch_size*2, x_height, x_width, total_time_steps]),
                                                 output_types=tf.float64)
        batched_x = dataset.batch(batch_size * 2).repeat(epochs)
    elif dname == "mmnist":
        data_path = "../data/mmnist/mnist_training_set.npy"
        training_data = np.load(data_path) / 255.0
        training_data = tf.transpose(training_data, (1, 0, 2, 3))
        training_data = tf.transpose(training_data, (0, 2, 1, 3))
        dataset = tf.data.Dataset.from_tensor_slices(training_data)
        batched_x = dataset.batch(batch_size * 2).repeat(epochs)
    elif dname == "mazes":
        # path to data
        root_path = '../data/'
        data_reader = data_utils.DataReader(dataset=dname, time_steps=total_time_steps,
                                            root=root_path, custom_frame_size=x_height)
        batched_x = data_reader.provide_dataset(batch_size=batch_size*2)

    # Create instances of generator, discriminator_h and
    # discriminator_m CONV VERSION
    encode_period = [int(x) for x in args.enc_period.split(",")]
    decode_period = [int(x) for x in args.dec_period.split(",")]
    z_height = x_height // 16
    z_width = x_width // 16

    # Define a standard multivariate normal for (z1, z2, ..., zT) --> (y1, y2, ..., yT)
    dist_z = tfp.distributions.Normal(0.0, 1.0)

    generator_type = args.generator
    if generator_type == 'convLSTM' or generator_type == 'reg_convLSTM':
        encoder = gan.VideoEncoderConvLSTM(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width, x_height,
                                           z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels,
                                           dropout=dp, rnn_dropout=rnn_dp, reg=regularization, cw=cw, period=encode_period)
        decoder = gan.VideoDecoderConvLSTM(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width, x_height,
                                           z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels,
                                           dropout=dp, rnn_dropout=rnn_dp, output_activation=g_output_activation,
                                           reg=regularization, cw=cw, period=decode_period)
    elif generator_type == 'lstm+conv':
        encoder = gan.VideoEncoder(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width, x_height,
                                     z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels,
                                     dropout=dp, rnn_dropout=rnn_dp, period=encode_period)
        decoder = gan.VideoDecoder(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width, x_height,
                                   z_width, z_height, g_filter_size, bn=bn, nlstm=nlstm, nchannel=channels,
                                   dropout=dp, rnn_dropout=rnn_dp, output_activation=g_output_activation,
                                   period=decode_period)
    else:
        generator = gan.VideoAutoRegGenerator(batch_size, int_time_steps, pred_time_steps, g_state_size, x_width,
                                              x_height, z_width, z_height, filter_size=g_filter_size,
                                              bn=bn, nlstm=nlstm, nchannel=channels, generation_steps=generation_steps)

    discriminator_h = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                             z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)
    discriminator_m = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                             z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)
    '''
    discriminator_h_bc = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                                z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)
    discriminator_m_bc = gan.VideoDiscriminator(batch_size, total_time_steps, d_state_size, x_width, x_height, z_width,
                                                z_height, filter_size=d_filter_size, bn=bn, nchannel=channels)
    '''
    n_blocks = args.n_blocks

    quantiser = data_utils.Quantiser(n=n_blocks)

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

    f_name = "{}_{}_lr{}_hidden{}_lam{}_{}_{}blocks{}".format(dname, generator_type, gen_lr, g_filter_size, reg_penalty,
                                                            g_output_activation, n_blocks, args.ckpt_str)

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

    ag = True

    @tf.function(autograph=ag)
    def disc_training_step(real_in, real_in_p, real_pred, real_pred_p):
        if generator_type == 'lstm+conv':
            hidden_z = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height*z_width])
            hidden_z_p = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height*z_width])
        else:
            hidden_z = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height, z_width, 128])
            hidden_z_p = dist_z.sample([batch_size, pred_time_steps // decode_period[-1], z_height, z_width, 128])

        with tf.GradientTape() as disc_tape:
            features, lstm_h, lstm_c = encoder.call(real_in)
            inp = tf.repeat(features[-1][:, -1, :][:, tf.newaxis, :], pred_time_steps // decode_period[-1], axis=1)
            dec_inp = tf.concat([inp, hidden_z], axis=-1)

            fake_pred = decoder.call(dec_inp, lstm_h, lstm_c)

            real = tf.concat((real_in, real_pred), axis=2)
            fake = tf.concat((real_in, fake_pred), axis=2)
            # Quantisation
            if args.quantisation:
                vq_real = quantiser.quantise(real)
                vq_fake = quantiser.quantise(fake)
                # vq_h_fake = quantiser.quantise(h_fake)
                # vq_h_real = quantiser.quantise(h_real)
                # vq_m_real = quantiser.quantise(m_real)
                # vq_m_fake = quantiser.quantise(m_fake)
                h_fake = discriminator_h.call(vq_fake)
                h_real = discriminator_h.call(vq_real)

                m_real = discriminator_m.call(vq_real)
                m_fake = discriminator_m.call(vq_fake)
            else:
                h_fake = discriminator_h.call(fake)
                h_real = discriminator_h.call(real)

                m_real = discriminator_m.call(real)
                m_fake = discriminator_m.call(fake)

            if args.mixed_sinkhorn:
                features_p, lstm_h_p, lstm_c_p = encoder.call(real_in_p)
                inp_p = tf.repeat(features_p[-1][:, -1, :][:, tf.newaxis, :], pred_time_steps // decode_period[-1],
                                  axis=1)
                dec_inp_p = tf.concat([inp_p, hidden_z_p], axis=-1)
                fake_pred_p = decoder.call(dec_inp_p, lstm_h_p, lstm_c_p)
                real_p = tf.concat((real_in_p, real_pred_p), axis=2)
                fake_p = tf.concat((real_in_p, fake_pred_p), axis=2)

                # Quantisation
                if args.quantisation:
                    vq_real_p = quantiser.quantise(real_p)
                    vq_fake_p = quantiser.quantise(fake_p)
                    # vq_h_real_p = quantiser.quantise(h_real_p)
                    # vq_h_fake_p = quantiser.quantise(h_fake_p)
                    # vq_m_real_p = quantiser.quantise(m_real_p)
                    h_real_p = discriminator_h.call(vq_real_p)
                    h_fake_p = discriminator_h.call(vq_fake_p)
                    m_real_p = discriminator_m.call(vq_real_p)
                    loss = gan_utils.compute_mixed_sinkhorn_loss(vq_real, vq_fake, m_real, m_fake,h_fake, scaling_coef,
                                                                 sinkhorn_eps, sinkhorn_l, vq_real_p, vq_fake_p,
                                                                 m_real_p, h_real_p, h_fake_p, video=True)
                else:
                    h_real_p = discriminator_h.call(real_p)
                    h_fake_p = discriminator_h.call(fake_p)
                    m_real_p = discriminator_m.call(real_p)
                    loss = gan_utils.compute_mixed_sinkhorn_loss(real, fake, m_real, m_fake, h_fake, scaling_coef,
                                                                 sinkhorn_eps, sinkhorn_l, real_p, fake_p, m_real_p,
                                                                 h_real_p, h_fake_p, video=True)

            elif args.bi_causal:
                if args.quantisation:
                    loss = gan_utils.compute_sinkhorn_loss_bi_causal(vq_real, vq_fake, scaling_coef, sinkhorn_eps,
                                                                     sinkhorn_l, h_fake, m_real, h_real,
                                                                     m_fake, video=True)

                else:
                    loss = gan_utils.compute_sinkhorn_loss_bi_causal(real, fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                                                     h_fake, m_real, h_real, m_fake, video=True)

            else:
                if args.quantisation:
                    loss = gan_utils.compute_sinkhorn_loss_no_mix(vq_real, vq_fake, scaling_coef, sinkhorn_eps,
                                                                  sinkhorn_l, h_fake, m_real, h_real, m_fake, video=True)
                else:
                    loss = gan_utils.compute_sinkhorn_loss_no_mix(real, fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                                                  h_fake, m_real, h_real, m_fake, video=True)
            pm1 = gan_utils.scale_invariante_martingale_regularization(m_real, reg_penalty, scaling_coef)
            disc_loss = - loss + pm1
            # update discriminator parameters
        disch_grads, discm_grads = disc_tape.gradient(disc_loss,
                                                      [discriminator_h.trainable_variables,
                                                       discriminator_m.trainable_variables])
        dischm_optimiser.apply_gradients(zip(disch_grads, discriminator_h.trainable_variables))
        dischm_optimiser.apply_gradients(zip(discm_grads, discriminator_m.trainable_variables))
        return pm1

    @tf.function(autograph=ag)
    def gen_training_step(real_in, real_in_p, real_pred, real_pred_p):
        hidden_z = dist_z.sample([batch_size, generation_steps // decode_period[-1], z_height, z_width, 128])
        hidden_z_p = dist_z.sample([batch_size, generation_steps // decode_period[-1], z_height, z_width, 128])

        with tf.GradientTape() as gen_tape:
            features, lstm_h, lstm_c = encoder.call(real_in)
            inp = tf.repeat(features[-1][:, -1, :][:, tf.newaxis, :], pred_time_steps // decode_period[-1], axis=1)
            dec_inp = tf.concat([inp, hidden_z], axis=-1)

            fake_pred = decoder.call(dec_inp, lstm_h, lstm_c)

            real = tf.concat((real_in, real_pred), axis=2)
            fake = tf.concat((real_in, fake_pred), axis=2)

            # Quantisation
            if args.quantisation:
                vq_real = quantiser.quantise(real)
                vq_fake = quantiser.quantise(fake)
                h_fake = discriminator_h.call(vq_fake)
                h_real = discriminator_h.call(vq_real)

                m_real = discriminator_m.call(vq_real)
                m_fake = discriminator_m.call(vq_fake)
            else:
                h_fake = discriminator_h.call(fake)
                h_real = discriminator_h.call(real)

                m_real = discriminator_m.call(real)
                m_fake = discriminator_m.call(fake)

            if args.mixed_sinkhorn:
                features_p, lstm_h_p, lstm_c_p = encoder.call(real_in_p)
                inp_p = tf.repeat(features_p[-1][:, -1, :][:, tf.newaxis, :], pred_time_steps // decode_period[-1],
                                  axis=1)
                dec_inp_p = tf.concat([inp_p, hidden_z_p], axis=-1)
                fake_pred_p = decoder.call(dec_inp_p, lstm_h_p, lstm_c_p)
                real_p = tf.concat((real_in_p, real_pred_p), axis=2)
                fake_p = tf.concat((real_in_p, fake_pred_p), axis=2)

                # Quantisation
                if args.quantisation:
                    vq_real_p = quantiser.quantise(real_p)
                    vq_fake_p = quantiser.quantise(fake_p)
                    h_real_p = discriminator_h.call(vq_real_p)
                    h_fake_p = discriminator_h.call(vq_fake_p)
                    m_real_p = discriminator_m.call(vq_real_p)
                    loss1 = gan_utils.compute_mixed_sinkhorn_loss(
                        real, fake, m_real, m_fake, h_fake, scaling_coef,
                        sinkhorn_eps, sinkhorn_l, real_p, fake_p, m_real_p,
                        h_real_p, h_fake_p, video=True)
                    vq_loss = gan_utils.compute_mixed_sinkhorn_loss(vq_real, vq_fake, m_real, m_fake,
                                                                    h_fake, scaling_coef, sinkhorn_eps,
                                                                    sinkhorn_l, vq_real_p, vq_fake_p, m_real_p,
                                                                    h_real_p, h_fake_p, video=True)
                    loss = loss1 + tf.stop_gradient(vq_loss - loss1)

                else:
                    h_real_p = discriminator_h.call(real_p)
                    h_fake_p = discriminator_h.call(fake_p)
                    m_real_p = discriminator_m.call(real_p)
                    loss = gan_utils.compute_mixed_sinkhorn_loss(real, fake, m_real, m_fake, h_fake, scaling_coef,
                                                                 sinkhorn_eps, sinkhorn_l, real_p, fake_p, m_real_p,
                                                                 h_real_p, h_fake_p, video=True)
                if generator_type == "reg_convLSTM":
                    pullaway = gan_utils.pullaway_loss(features[-1])
                    pullaway_p = gan_utils.pullaway_loss(features_p[-1])

                    loss = loss + pullaway + pullaway_p

            elif args.bi_causal:
                if args.quantisation:
                    loss1 = gan_utils.compute_sinkhorn_loss_bi_causal(real, fake, scaling_coef, sinkhorn_eps,
                                                                      sinkhorn_l, h_fake, m_real, h_real, m_fake,
                                                                      video=True)
                    vq_loss = gan_utils.compute_sinkhorn_loss_bi_causal(vq_real, vq_fake, scaling_coef,
                                                                        sinkhorn_eps,
                                                                        sinkhorn_l, h_fake, m_real, h_real,
                                                                        m_fake, video=True)
                    loss = loss1 + tf.stop_gradient(vq_loss - loss1)
                else:
                    loss = gan_utils.compute_sinkhorn_loss_bi_causal(real, fake, scaling_coef, sinkhorn_eps,
                                                                     sinkhorn_l, h_fake, m_real, h_real, m_fake,
                                                                     video=True)
                if generator_type == "reg_convLSTM":
                    pullaway = gan_utils.pullaway_loss(features[-1])

                    loss = loss + pullaway

            else:
                if args.quantisation:
                    loss1 = gan_utils.compute_sinkhorn_loss_no_mix(real, fake, scaling_coef, sinkhorn_eps,
                                                                   sinkhorn_l, h_fake, m_real, h_real, m_fake,
                                                                   video=True)
                    vq_loss = gan_utils.compute_sinkhorn_loss_no_mix(vq_real, vq_fake, scaling_coef, sinkhorn_eps,
                                                                     sinkhorn_l, h_fake, m_real, h_real,
                                                                     m_fake, video=True)
                    loss = loss1 + tf.stop_gradient(vq_loss - loss1)
                else:
                    loss = gan_utils.compute_sinkhorn_loss_no_mix(real, fake, scaling_coef, sinkhorn_eps,
                                                                  sinkhorn_l, h_fake, m_real, h_real, m_fake,
                                                                  video=True)
                if generator_type == "reg_convLSTM":
                    pullaway = gan_utils.pullaway_loss(features[-1])

                    loss = loss + pullaway
                    # update encoder and decoder parameters
        enc_grads, dec_grads = gen_tape.gradient(loss, [encoder.trainable_variables, decoder.trainable_variables])
        gen_optimiser.apply_gradients(zip(enc_grads, encoder.trainable_variables))
        gen_optimiser.apply_gradients(zip(dec_grads, decoder.trainable_variables))
        return loss

    with tqdm.trange(epochs, ncols=100, unit="epoch") as ep:
        for _ in ep:
            it = tqdm.tqdm(ncols=100)
            for x in batched_x:
                if x.shape[0] != batch_size*2:
                    continue
                it_counts += 1
                # split the batches for x and x'
                real_data = x[0:batch_size, ]
                real_data_p = x[batch_size:, ]

                real_data = tf.reshape(real_data, [batch_size, x_height, total_time_steps, x_width, -1])
                real_data_p = tf.reshape(real_data_p, [batch_size, x_height, total_time_steps, x_width, -1])

                # throw away alpha channel
                real_data = tf.cast(real_data[..., :channels], tf.float32)
                real_data_p = tf.cast(real_data_p[..., :channels], tf.float32)

                # split real data to training inputs and predictions
                real_inputs = real_data[:, :, :int_time_steps, :, :]
                real_inputs_p = real_data_p[:, :, :int_time_steps, :, :]
                real_preds = real_data[:, :, int_time_steps:, :, :]
                real_preds_p = real_data_p[:, :, int_time_steps:, :, :]

                pm = disc_training_step(real_inputs, real_inputs_p, real_preds, real_preds_p)
                loss = gen_training_step(real_inputs, real_inputs_p, real_preds, real_preds_p)
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
                        if generator_type == 'reg_convLSTM' or generator_type == 'convLSTM' \
                                or generator_type == 'lstm+conv':
                            # save model to file
                            encoder.save_weights("./trained/{}/{}_encoder/".format(test, model_fn))
                            decoder.save_weights("./trained/{}/{}_decoder/".format(test, model_fn))
                            discriminator_h.save_weights("./trained/{}/{}_h/".format(test, model_fn))
                            discriminator_m.save_weights("./trained/{}/{}_m/".format(test, model_fn))

                            if it_counts % 10000 == 0:
                                encoder.save_weights("./trained/{}/{}_iter{}_encoder/".format(test, model_fn, it_counts))
                                decoder.save_weights("./trained/{}/{}_iter{}_decoder/".format(test, model_fn, it_counts))
                                discriminator_h.save_weights("./trained/{}/{}_iter{}_h/".format(test, model_fn, it_counts))
                                discriminator_m.save_weights("./trained/{}/{}_iter{}_m/".format(test, model_fn, it_counts))
                            # load model
                            # encoder_generation.load_weights("./trained/{}/{}_encoder/".format(test, model_fn))
                            # decoder_generation.load_weights("./trained/{}/{}_decoder/".format(test, model_fn))
                            prediction_loops = 2
                            preditions = []
                            inputs = real_inputs
                            for i in range(prediction_loops):
                                if generator_type == 'lstm+conv':
                                    hidden_z = dist_z.sample(
                                        [batch_size, pred_time_steps // decode_period[-1], z_height * z_width])
                                else:
                                    hidden_z = dist_z.sample(
                                        [batch_size, pred_time_steps // decode_period[-1], z_height, z_width, 128])
                                features, lstm_h, lstm_c = encoder.call(real_inputs, training=False)
                                inp = tf.repeat(features[-1][:, -1, :][:, tf.newaxis, :],
                                                pred_time_steps // decode_period[-1], axis=1)
                                dec_inp = tf.concat([inp, hidden_z], axis=-1)
                                preds = decoder.call(dec_inp, lstm_h, lstm_c)
                                preditions.append(preds)
                                real_inputs = preds[:, :, (pred_time_steps-int_time_steps):, :, :]

                            preds = np.asarray(preditions)
                            preds = np.transpose(preds, (1, 0, 2, 3, 4, 5))
                            preds = np.transpose(preds, (0, 2, 1, 3, 4, 5))
                            preds = preds.reshape(batch_size, x_height, pred_time_steps * prediction_loops,
                                                  x_width, channels)

                        else:
                            # save model to file
                            generator.save_weights("./trained/{}/{}/".format(test, model_fn))
                            discriminator_h.save_weights("./trained/{}/{}_h/".format(test, model_fn))
                            discriminator_m.save_weights("./trained/{}/{}_m/".format(test, model_fn))
                            # load model
                            generator.load_weights("./trained/{}/{}/".format(test, model_fn))
                            preds = generator.call(hidden_z, real_inputs, training=False)

                        images = tf.concat((inputs, preds), axis=2)
                        images = tf.reshape(images, [batch_size, x_height,
                                                     x_width * (int_time_steps+pred_time_steps*prediction_loops),
                                                     channels])
                        # plot first 10 samples within one image
                        img = tf.concat(list(images[:min(10, batch_size)]), axis=0)[None]
                        with train_writer.as_default():
                            tf.summary.image("Training data", img, step=it_counts)

                continue
    print("--- The entire training takes %s minutes ---" % ((time.time() - start_time) / 60.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cot')

    parser.add_argument('-d', '--dname', type=str, default='mmnist',
                        choices=['animation', 'human_action', 'ucf', 'kth', 'penn_action', 'mmnist', 'mazes'])
    parser.add_argument('-t', '--test',  type=str, default='cot', choices=['cot'])
    parser.add_argument('-gen', '--generator', type=str, default='reg_convLSTM', choices=['convLSTM', 'lstm+conv',
                                                                                      'arconvlstm', 'reg_convLSTM'])
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-gss', '--g_state_size', type=int, default=8)
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=8)
    parser.add_argument('-dss', '--d_state_size', type=int, default=8)
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=8)
    # animation data has T=13 and human action data has T=16
    parser.add_argument('-tts', '--total_time_steps', type=int, default=20)
    parser.add_argument('-its', '--int_time_steps', type=int, default=8)
    parser.add_argument('-gts', '--gen_time_steps', type=int, default=12)
    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
    parser.add_argument('-reg_p', '--reg_penalty', type=float, default=1.0)
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
    parser.add_argument('-g', '--gen', type=str, default="fc", choices=["lstm", "fc"])
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-p', '--path', type=str, default='../data/animation/*.tfrecord')
    parser.add_argument('-save', '--save_freq', type=int, default=100)
    parser.add_argument('-lr', '--lr', type=float, default=5e-4)
    parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
    parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
    parser.add_argument('-nch', '--n_channels', type=int, default=1)
    parser.add_argument('-dp', '--dropout', type=float, default=0.0)
    parser.add_argument('-rdp', '--rnn_dropout', type=float, default=0.0)
    parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
    # Scale parameter applied will be 1.0 / scaling_coef
    parser.add_argument('-sc', '--scaling_coef', type=float, default=20.0)
    parser.add_argument('-mix', '--mixed_sinkhorn', type=bool, default=False)
    parser.add_argument('-ckpt', '--checkpoint', type=bool, default=False)
    parser.add_argument('-cn', '--ckpt_name', type=str, default='mmnist_convLSTM_lr0.001_hidden8_lam1.0_sigmoid_3blocks-no_mix')
    parser.add_argument('-bc', '--bi_causal', type=bool, default=False)
    parser.add_argument('-vq', '--quantisation', type=bool, default=True)
    parser.add_argument('-cw', '--clockwork', type=bool, default=False)
    parser.add_argument('-regu', '--regularization', type=bool, default=True)
    parser.add_argument('-nbs', '--n_blocks', type=int, default=2)
    parser.add_argument('-ct', '--ckpt_str', type=str, default='-full_training')
    # animation, human action and ucf have shape [64, 64]
    # kth has shape [120, 160]
    # penn action has shape [270, 360]
    parser.add_argument('-xh', '--height', type=int, default=64)
    parser.add_argument('-xw', '--width', type=int, default=64)
    parser.add_argument('-ne', '--n_epochs', type=int, default=100)
    parser.add_argument('-epd', '--enc_period', type=str, default="1,1,1,1")
    parser.add_argument('-dpd', '--dec_period', type=str, default="1,1,1,1")

    args = parser.parse_args()

    train(args)
