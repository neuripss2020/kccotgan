from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx('float32')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


class RNNEncoder(tf.keras.Model):
    '''
        Encoder that uses RNN for creating features for time series inputs (x_1, x_2,...,x_T)
        Args:
            inputs: ..
        Returns:
            features as outputs of an encoder
    '''

    def __init__(self, batch_size, int_time_steps, Dx, state_size, rnn_bn=False):
        super().__init__()

        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.int_time_steps = int_time_steps
        self.rnn_bn = rnn_bn
        #
        # encoder
        #
        self.encoder1 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)

        if self.rnn_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.encoder2 = tf.keras.layers.LSTM(self.state_size // 2, return_state=True, return_sequences=True)

        if self.rnn_bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        self.encoder3 = tf.keras.layers.LSTM(self.state_size // 4, return_state=True, return_sequences=True)

    def call(self, input_real, training=True, mask=None):
        x = tf.reshape(tensor=input_real, shape=[self.batch_size, self.int_time_steps, self.Dx])
        # LSTM encoder
        output_seq1, _, _ = self.encoder1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1)
        output_seq2, _, _ = self.encoder2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2)
        features, input_h1, input_c1 = self.encoder3(output_seq2, training=training)
        return features, input_h1, input_c1


class RNNDecoder(tf.keras.Model):
    '''
    Decoder that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_K)
    conditioned on the latent variable Z  and input time series (x_1, x_2, ..., x_T).
    Args:
        inputs: ...
    Returns:
        output of generator
    '''

    def __init__(self, batch_size, pred_time_steps, Dz, Dx, state_size, filter_size,
                 output_activation='sigmoid', bn=False, nlayer=2, rnn_bn=False):
        super().__init__()

        self.Dz = Dz
        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.pred_time_steps = pred_time_steps
        self.out_activation = output_activation
        self.rnn_bn = rnn_bn
        #
        # decoder
        #
        self.decoder1 = tf.keras.layers.LSTM(self.state_size // 4,  return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn3 = tf.keras.layers.BatchNormalization()
        self.decoder2 = tf.keras.layers.LSTM(self.state_size // 2, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn4 = tf.keras.layers.BatchNormalization()
        self.decoder3 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn5 = tf.keras.layers.BatchNormalization()

        # Dense layers
        self.fc = tf.keras.Sequential()
        for i in range(nlayer - 1):
            self.fc.add(tf.keras.layers.Dense(units=filter_size, activation=None, use_bias=True))
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(tf.keras.layers.ReLU())
        self.fc.add(tf.keras.layers.Dense(units=Dx, activation=self.out_activation, use_bias=True))

    def call(self, input_z, enc_h, enc_c, training=True, mask=None):
        z = tf.reshape(tensor=input_z, shape=[self.batch_size, self.pred_time_steps, self.Dz])
        # LSTM + FC decoder
        output_seq4, _, _ = self.decoder1([z, enc_h, enc_c], training=training)
        if self.rnn_bn:
            output_seq4 = self.bn3(output_seq4)
        output_seq5, _, _ = self.decoder2(output_seq4, training=training)
        if self.rnn_bn:
            output_seq5 = self.bn4(output_seq5)
        output_seq6, _, _ = self.decoder3(output_seq5, training=training)
        if self.rnn_bn:
            output_seq6 = self.bn5(output_seq6)
        y = self.fc(output_seq6)
        x = tf.reshape(tensor=y, shape=[self.batch_size, self.pred_time_steps, self.Dx])
        return x


class RNNGenerator(tf.keras.Model):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z.
    Args:
        inputs: (numpy array) latent variables as inputs to the RNN model has shape
                [batch_size, time_step, sub_sequence_hidden_dims]
    Returns:
        output of generator
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, Dz, Dx, state_size, filter_size,
                 output_activation='sigmoid', bn=False, nlayer=2, rnn_bn=False):
        super().__init__()

        self.Dz = Dz
        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.total_time_steps = int(int_time_steps + pred_time_steps)
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.out_activation = output_activation
        self.rnn_bn = rnn_bn
        self.nlayer = nlayer
        #
        # encoder
        #
        self.encoder1 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)

        if self.rnn_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.encoder2 = tf.keras.layers.LSTM(self.state_size // 2, return_state=True, return_sequences=True)

        if self.rnn_bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        self.encoder3 = tf.keras.layers.LSTM(self.state_size // 4, return_state=True, return_sequences=True)
        #
        # decoder
        #
        self.decoder1 = tf.keras.layers.LSTM(self.state_size // 4,  return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn3 = tf.keras.layers.BatchNormalization()
        self.decoder2 = tf.keras.layers.LSTM(self.state_size // 2, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn4 = tf.keras.layers.BatchNormalization()
        self.decoder3 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn5 = tf.keras.layers.BatchNormalization()

        # Dense layers
        self.fc = tf.keras.Sequential()
        for i in range(self.nlayer - 1):
            self.fc.add(tf.keras.layers.Dense(units=filter_size, activation=None, use_bias=True))
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(tf.keras.layers.ReLU())
        self.fc.add(tf.keras.layers.Dense(units=Dx, activation=self.out_activation, use_bias=True))

    def call(self, input_z, input_real, training=True, mask=None):
        x = tf.reshape(tensor=input_real, shape=[self.batch_size, self.int_time_steps, self.Dx])
        z = tf.reshape(tensor=input_z, shape=[self.batch_size, self.pred_time_steps, self.Dz])
        # LSTM encoder
        output_seq1, _, _ = self.encoder1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1)
        output_seq2, _, _ = self.encoder2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2)
        output_seq3, input_h1, input_c1 = self.encoder3(output_seq2, training=training)
        # LSTM + FC decoder
        # padding = tf.zeros([self.batch_size, self.pred_time_steps-1, self.Dx + self.Dz], tf.float32)
        # inp = tf.concat([z[:, 0, :], x[:, -1, :]], axis=-1)
        # inp = tf.concat([tf.expand_dims(inp, axis=1), padding], axis=1)
        output_seq4, _, _ = self.decoder1([z, input_h1, input_c1], training=training)
        if self.rnn_bn:
            output_seq4 = self.bn3(output_seq4)
        output_seq5, _, _ = self.decoder2(output_seq4, training=training)
        if self.rnn_bn:
            output_seq5 = self.bn4(output_seq5)
        output_seq6, _, _ = self.decoder3(output_seq5, training=training)
        if self.rnn_bn:
            output_seq6 = self.bn5(output_seq6)
        y = self.fc(output_seq6)
        x = tf.reshape(tensor=y, shape=[self.batch_size, self.pred_time_steps, self.Dx])
        return x


class RNNDiscriminator(tf.keras.Model):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, Dz, Dx, state_size, bn=False,
                 output_activation='sigmoid'):
        super().__init__()

        self.batch_size = batch_size
        self.state_size = state_size
        self.Dz = Dz
        self.Dx = Dx
        self.total_time_steps = int(int_time_steps + pred_time_steps)
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.out_activation = output_activation
        self.rnn_bn = bn

        self.lstm1 = tf.keras.layers.LSTM(self.state_size // 8, return_sequences=True)
        if self.rnn_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.lstm2 = tf.keras.layers.LSTM(self.state_size // 4, return_sequences=True)
        if self.rnn_bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        self.lstm3 = tf.keras.layers.LSTM(self.state_size // 2, return_sequences=True)
        if self.rnn_bn:
            self.bn3 = tf.keras.layers.BatchNormalization()

        self.lstm4 = tf.keras.layers.LSTM(self.state_size // 2, return_sequences=True)
        if self.rnn_bn:
            self.bn4 = tf.keras.layers.BatchNormalization()

        self.lstm5 = tf.keras.layers.LSTM(self.state_size // 4, return_sequences=True)
        if self.rnn_bn:
            self.bn5 = tf.keras.layers.BatchNormalization()

        self.lstm6 = tf.keras.layers.LSTM(self.state_size // 8, return_sequences=True, activation=self.out_activation)

    def call(self, inputs, training=True, mask=None):
        x = tf.reshape(tensor=inputs, shape=[self.batch_size, self.total_time_steps, self.Dx])
        output_seq1 = self.lstm1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1)
        output_seq2 = self.lstm2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2)
        output_seq3 = self.lstm3(output_seq2, training=training)
        if self.rnn_bn:
            output_seq3 = self.bn3(output_seq3)
        output_seq4 = self.lstm4(output_seq3, training=training)
        if self.rnn_bn:
            output_seq4 = self.bn4(output_seq4)
        output_seq5 = self.lstm5(output_seq4, training=training)
        if self.rnn_bn:
            output_seq5 = self.bn5(output_seq5)
        output_seq6 = self.lstm6(output_seq5, training=training)
        return output_seq6


class ToyGenerator(tf.keras.Model):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z.
    Args:
        inputs: (numpy array) latent variables as inputs to the RNN model has shape
                [batch_size, time_step, sub_sequence_hidden_dims]
    Returns:
        output of generator
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, Dz, Dx, state_size, filter_size,
                 output_activation='sigmoid', bn=False, nlstm=1, nlayer=2, rnn_bn=False):
        super().__init__()

        self.Dz = Dz
        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.total_time_steps = int(int_time_steps + pred_time_steps)
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.out_activation = output_activation
        self.rnn_bn = rnn_bn
        self.nlayer = nlayer
        #
        # encoder
        #
        self.encoder1 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.encoder2 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        self.encoder3 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn3 = tf.keras.layers.BatchNormalization()
        #
        # decoder
        #
        self.decoder1 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn4 = tf.keras.layers.BatchNormalization()
        self.decoder2 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)
        if self.rnn_bn:
            self.bn5 = tf.keras.layers.BatchNormalization()
        self.decoder3 = tf.keras.layers.LSTM(self.state_size, return_state=True, return_sequences=True)

        # Dense layers
        self.fc = tf.keras.Sequential()
        for i in range(self.nlayer-1):
            self.fc.add(tf.keras.layers.Dense(units=filter_size, activation=None, use_bias=True))
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(tf.keras.layers.ReLU())
        self.fc.add(tf.keras.layers.Dense(units=Dx, activation=self.out_activation, use_bias=True))

    def call(self, input_z, input_real, training=True, mask=None):
        x = tf.reshape(tensor=input_real, shape=[self.batch_size, self.int_time_steps, self.Dx])
        z = tf.reshape(tensor=input_z, shape=[self.batch_size, self.pred_time_steps, self.Dz])

        # LSTM encoder
        output_seq1, _, _ = self.encoder1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1)
        output_seq2, _, _ = self.encoder2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2)
        output_seq3, last_h, last_c = self.encoder3(output_seq2, training=training)
        if self.rnn_bn:
            output_seq3 = self.bn3(output_seq3)

        # LSTM + FC decoder
        y = tf.zeros([self.batch_size, 1, self.Dx])
        for n in range(1, self.pred_time_steps + 1):
            inp = tf.concat([z[:, n - 1, :], last_h, last_c], axis=-1)
            inp = tf.reshape(inp, [self.batch_size, 1, -1])
            output_seq4, _, _ = self.decoder1(inp, training=training)
            if self.rnn_bn:
                output_seq4 = self.bn4(output_seq4)
            output_seq5, _, _ = self.decoder2(output_seq4, training=training)
            if self.rnn_bn:
                output_seq5 = self.bn5(output_seq5)
            output_seq6, last_h, last_c = self.decoder3(output_seq5, training=training)
            inp_y = tf.concat([y[:, n - 1, :], last_h, last_c], axis=-1)
            inp_y = tf.reshape(inp_y, [self.batch_size, 1, -1])
            yt = self.fc(inp_y)
            y = tf.concat([y, yt], axis=1)
            '''
            inp = tf.concat([z[:, n - 1, :], last_h, last_c], axis=-1)
            inp = tf.reshape(inp, [self.batch_size, 1, -1])
            last_h = self.decoder1(inp, training=training)
            inp_y = tf.concat([y[:, n - 1, :], last_h, last_c], axis=-1)
            inp_y = tf.reshape(inp_y, [self.batch_size, 1, -1])
            yt = self.fc(inp_y, training=training)
            y = tf.concat([y, yt], axis=1)
            '''
        x = tf.reshape(tensor=y[:, 1:, :], shape=[self.batch_size, self.pred_time_steps, self.Dx])
        return x


class ToyDiscriminator(tf.keras.Model):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, Dz, Dx, state_size, filter_size, bn=False, kernel_size=5, strides=1,
                 output_activation="tanh", nlayer=3, nlstm=0):
        super().__init__()

        self.batch_size = batch_size
        self.state_size = state_size
        self.Dz = Dz
        self.Dx = Dx
        self.total_time_steps = int(int_time_steps + pred_time_steps)
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps

        self.fc = tf.keras.Sequential()
        self.fc.add(tf.keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size,
                                           padding="causal", strides=strides))

        for i in range(nlayer - 1):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(tf.keras.layers.ReLU())
            self.fc.add(tf.keras.layers.Conv1D(filters=state_size, kernel_size=kernel_size,
                                               activation=output_activation if i == nlayer - 2 else None,
                                               padding="causal", strides=strides))
        for i in range(nlstm):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(tf.keras.layers.LSTM(state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):
        x = tf.reshape(tensor=inputs, shape=[self.batch_size, self.total_time_steps, self.Dx])
        z = self.fc(x)
        return z


class VideoEncoder(tf.keras.Model):
    '''
    Encoder for creating latent representation (h_1, h_2,...,h_T) from the input video sequence (x_1, ..., x_T).
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, state_size, x_width, x_height, z_width=5,
                 z_height=5, filter_size=64, bn=False, nlstm=1, cat=False, nchannel=3, dropout=0.0, rnn_dropout=0.0,
                 period=[1, 2, 4]):
        super(VideoEncoder, self).__init__()
        self.batch_size = batch_size
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.nchannel = nchannel
        self.rnn_bn = bn
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.period = np.asarray(sorted(period))
        #
        # Conv encoder
        #
        self.conv_model = tf.keras.Sequential()
        self.conv_model.add(tf.keras.layers.Conv2D(self.filter_size, (5, 5), strides=(2, 2), padding='same',
                                                   input_shape=[self.x_height, self.x_width, self.nchannel]))

        self.conv_model.add(tf.keras.layers.BatchNormalization())
        self.conv_model.add(tf.keras.layers.Conv2D(self.filter_size * 2, (5, 5), strides=(2, 2), padding='same',
                                                   input_shape=[self.x_height, self.x_width, self.nchannel]))
        self.conv_model.add(tf.keras.layers.BatchNormalization())
        self.conv_model.add(tf.keras.layers.Conv2D(self.filter_size * 4, (5, 5), strides=(2, 2), padding='same',
                                                   input_shape=[self.x_height, self.x_width, self.nchannel]))
        self.conv_model.add(tf.keras.layers.BatchNormalization())

        self.encoder1 = tf.keras.layers.LSTM(self.filter_size * 64, return_sequences=True, return_state=True)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.encoder2 = tf.keras.layers.LSTM(self.filter_size * 32, return_sequences=True, return_state=True)
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.encoder3 = tf.keras.layers.LSTM(self.filter_size * 16, return_sequences=True, return_state=True)
        self.bn6 = tf.keras.layers.BatchNormalization()

    def call_all(self, inputs_real, training=True):
        # x input sequence
        input_shape = [self.batch_size, self.x_height, self.int_time_steps, self.x_width, self.nchannel]
        x = tf.reshape(tensor=inputs_real, shape=input_shape)
        x = tf.transpose(x, (0, 2, 1, 3, 4))
        x = tf.reshape(tensor=x, shape=[self.batch_size*self.int_time_steps, self.x_height, self.x_width, self.nchannel])
        conv_features = self.conv_model(x)
        rnn_inp = tf.reshape(conv_features, [self.batch_size, self.int_time_steps, -1])

        layer_outputs = []
        layer_h = []
        layer_c = []

        # first period/lag, e.g., k=1 for no skipping
        cwrnn1 = RNNClockworkLayer(rnn_inp.shape, rnn_layer=self.encoder1, k=self.period[0])
        output_seq1, input_h, input_c = cwrnn1.call(rnn_inp, training=True)

        # return the last state that contains all information about the input sequence
        # output_seq1, input_h, input_c = self.encoder1(x, training=training)
        layer_outputs.append(output_seq1)
        layer_h.append(input_h)
        layer_c.append(input_c)
        if self.rnn_bn:
            output_seq1 = self.bn4(output_seq1, training=training)

        cwrnn2 = RNNClockworkLayer(rnn_inp.shape, rnn_layer=self.encoder2, k=self.period[1],
                                   state_shape=output_seq1.shape)
        output_seq2, input_h, input_c = cwrnn2.call(rnn_inp, output_seq1, training=True)
        layer_outputs.append(output_seq2)
        layer_h.append(input_h)
        layer_c.append(input_c)
        if self.rnn_bn:
            output_seq2 = self.bn5(output_seq2, training=training)

        cwrnn3 = RNNClockworkLayer(rnn_inp.shape, rnn_layer=self.encoder3, k=self.period[2],
                                   state_shape=output_seq2.shape)
        output_seq3, input_h, input_c = cwrnn3.call(rnn_inp, output_seq2, training=True)
        layer_outputs.append(output_seq3)
        layer_h.append(input_h)
        layer_c.append(input_c)

        return layer_outputs, layer_h, layer_c

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)


class VideoDecoder(tf.keras.Model):
    '''
    Decoder for creating fake video sequence (y_1, y_2,...,y_T) from the latent variable Z
    and last hidden state from encoder.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, state_size, x_width, x_height, z_width=5,
                 z_height=5, filter_size=64, bn=False, output_activation="sigmoid", nlstm=1,
                 cat=False, nchannel=3, dropout=0.0, rnn_dropout=0.0, period=[1, 2, 4]):
        super(VideoDecoder, self).__init__()
        self.batch_size = batch_size
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.nchannel = nchannel
        self.rnn_bn = bn
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.period = np.asarray(sorted(period, reverse=True))
        self.factor1 = self.period[0] // self.period[1]
        self.factor2 = self.period[1] // self.period[2]
        self.factor3 = 1
        self.period_time_steps = self.pred_time_steps // self.period

        # compute paddings
        if self.x_height == self.x_width:
            stride4 = [2, 2]
            k_size4 = [6, 6]

            stride3 = [2, 2]
            k_size3 = [6, 6]

            stride2 = [2, 2]
            k_size2 = [4, 4]

            stride1 = [2, 2]
            k_size1 = [2, 2]
        elif self.x_height < self.x_width:
            stride3 = [2, 2]
            k_size3 = [6, 7]

            stride2 = [2, 2]
            k_size2 = [6, 7]

            stride1 = [2, 2]
            k_size1 = [6, 7]
        else:
            stride4 = [3, 2]
            k_size4 = [9, 8]

            stride3 = [3, 2]
            k_size3 = [7, 6]

            stride2 = [3, 2]
            k_size2 = [7, 6]

            stride1 = [3, 2]
            k_size1 = [7, 6]
        #
        # decoder
        #

        self.decoder1 = tf.keras.layers.LSTM(self.filter_size * 16, return_sequences=True, return_state=True)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.decoder2 = tf.keras.layers.LSTM(self.filter_size * 32, return_sequences=True, return_state=True)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.decoder3 = tf.keras.layers.LSTM(self.filter_size * 64, return_sequences=True, return_state=True)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv_model = tf.keras.Sequential()

        self.conv_model.add(tf.keras.layers.Conv2DTranspose(self.filter_size * 4, k_size1, strides=stride1,
                                                               padding='same',
                                                               use_bias=False, activation=None,
                                                               data_format='channels_last'))
        self.conv_model.add(tf.keras.layers.BatchNormalization())

        self.conv_model.add(tf.keras.layers.Conv2DTranspose(self.filter_size * 2, k_size2, strides=stride2, padding='same',
                                                        use_bias=False, activation=None,
                                                        data_format='channels_last'))

        self.conv_model.add(tf.keras.layers.BatchNormalization())

        self.conv_model.add(tf.keras.layers.Conv2DTranspose(self.filter_size, k_size3, strides=stride3, padding='same',
                                                        use_bias=False, activation=None,
                                                        data_format='channels_last'))

        self.conv_model.add(tf.keras.layers.BatchNormalization())

        self.conv_model.add(tf.keras.layers.Conv2DTranspose(self.nchannel, k_size4, strides=stride4, padding='same',
                                                            use_bias=False, activation=output_activation,
                                                            data_format='channels_last'))

    def computing_padding(self, h_out, h_in, k_size, stride):
        p1 = ((h_in[0] - 1) * stride[0] - h_out[0] + k_size[0]) / 2
        p2 = ((h_in[1] - 1) * stride[1] - h_out[1] + k_size[1]) / 2
        padding_h = int(abs(tf.math.ceil(p1)))
        padding_w = int(abs(tf.math.ceil(p2)))
        return [padding_h, padding_w]

    def call_all(self, inputs_z, input_h, input_c, training=True):
        # z has shape of [batch_size, time_step, sub_sequence_hidden_dims]
        z = tf.reshape(tensor=inputs_z, shape=[self.batch_size, self.period_time_steps[0], -1])

        # first period/lag, e.g., k=1 for no skipping
        cwrnn1 = RNNClockworkLayer(z.shape, self.decoder1, k=self.period[0], decoder=True, factor=self.factor1)
        output_seq1, h, c = cwrnn1.call([z, input_h[2], input_c[2]], training=True)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1, training=training)

        cwrnn2 = RNNClockworkLayer(output_seq1.shape, self.decoder2, k=self.period[1],
                                   decoder=True, factor=self.factor2)
        output_seq2, h, c = cwrnn2.call([output_seq1, input_h[1], input_c[1]], training=True)

        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2, training=training)

        cwrnn3 = RNNClockworkLayer(output_seq2.shape, self.decoder3, k=self.period[2],
                                   decoder=True, factor=self.factor3)
        output_seq3, h, c = cwrnn3.call([output_seq2, input_h[0], input_c[0]], training=True)

        if self.rnn_bn:
            output_seq3 = self.bn3(output_seq3, training=training)

        conv_inputs3 = tf.reshape(output_seq3, [self.batch_size * self.pred_time_steps, 4, 4, -1])

        y = self.conv_model(conv_inputs3)

        y = tf.reshape(y, [self.batch_size, self.pred_time_steps, self.x_height, self.x_width, self.nchannel])
        y = tf.transpose(y, (0, 2, 1, 3, 4))
        return y

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)


class VideoEncoderConvLSTM(tf.keras.Model):
    '''
    Encoder for creating latent representation (h_1, h_2,...,h_T) from the input video sequence (x_1, ..., x_T).
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, state_size, x_width, x_height, z_width=5,
                 z_height=5, filter_size=64, bn=False, nlstm=1, cat=False, nchannel=3, dropout=0.0, rnn_dropout=0.0,
                 reg=False, cw=False, period=[1, 2, 4]):
        super(VideoEncoderConvLSTM, self).__init__()
        self.batch_size = batch_size
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.cw = cw
        self.nchannel = nchannel
        self.rnn_bn = bn
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.period = np.asarray(sorted(period))
        if reg:
            self.reg = tf.keras.regularizers.L2(0.01)
        else:
            self.reg = None
        #
        # ConvLSTM encoder
        #
        self.conv_embedding1 = tf.keras.layers.Conv2D(self.filter_size, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[self.x_height, self.x_width, self.nchannel],
                                                      kernel_regularizer=self.reg, bias_regularizer=self.reg,
                                                      activity_regularizer=self.reg)

        self.encoder1 = tf.keras.layers.ConvLSTM2D(self.filter_size * 2, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                                   data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv_embedding2 = tf.keras.layers.Conv2D(self.filter_size * 2, (5, 5), strides=(2, 2), padding='same',
                                                      input_shape=[self.x_height, self.x_width, self.nchannel],
                                                      kernel_regularizer=self.reg, bias_regularizer=self.reg,
                                                      activity_regularizer=self.reg)

        self.encoder2 = tf.keras.layers.ConvLSTM2D(self.filter_size * 4, kernel_size=(5, 5), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv_embedding3 = tf.keras.layers.Conv2D(self.filter_size * 4, (5, 5), strides=(4, 4), padding='same',
                                                      input_shape=[self.x_height, self.x_width, self.nchannel],
                                                      kernel_regularizer=self.reg, bias_regularizer=self.reg,
                                                      activity_regularizer=self.reg)

        self.encoder3 = tf.keras.layers.ConvLSTM2D(self.filter_size * 8, kernel_size=(5, 5), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)

        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv_embedding4 = tf.keras.layers.Conv2D(self.filter_size * 8, (5, 5), strides=(8, 8), padding='same',
                                                      input_shape=[self.x_height, self.x_width, self.nchannel],
                                                      kernel_regularizer=self.reg, bias_regularizer=self.reg,
                                                      activity_regularizer=self.reg)

        self.encoder4 = tf.keras.layers.ConvLSTM2D(self.filter_size * 16, kernel_size=(5, 5), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)

    def call_all(self, inputs_real, training=True):
        # x input sequence
        input_shape = [self.batch_size, self.x_height, self.int_time_steps, self.x_width, self.nchannel]
        x = tf.reshape(tensor=inputs_real, shape=input_shape)
        # x = tf.transpose(x, (0, 2, 1, 3, 4))
        layer_outputs = []
        layer_h = []
        layer_c = []

        # first period/lag, e.g., k=1 for no skipping
        # cwrnn1 = ClockworkLayer(input_shape, rnn_layer=self.encoder1, k=self.period[0],
        # embedding_layer=self.conv_embedding1)
        cwrnn1 = ClockworkLayer(input_shape, rnn_layer=self.encoder1, k=self.period[0], clockwork=self.cw)
        output_seq1, input_h, input_c = cwrnn1.call(x, training=True)
        layer_outputs.append(output_seq1)
        layer_h.append(input_h)
        layer_c.append(input_c)

        # return the last state that contains all information about the input sequence
        # output_seq1, input_h, input_c = self.encoder1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1, training=training)

        # output_seq1 = tf.transpose(output_seq1, [0, 2, 1, 3, 4])
        # output_seq2, input_h, input_c = ClockworkLayer(output_seq1.shape, rnn_layer=self.encoder2,
        #                                                k=self.period[1]).call(output_seq1, training=True)

        cwrnn2 = ClockworkLayer(input_shape, rnn_layer=self.encoder2, k=self.period[1],
                                embedding_layer=self.conv_embedding2, state_shape=output_seq1.shape, clockwork=self.cw)
        output_seq2, input_h, input_c = cwrnn2.call(x, output_seq1, training=True)
        layer_outputs.append(output_seq2)
        layer_h.append(input_h)
        layer_c.append(input_c)

        # output_seq2, input_h, input_c = self.encoder2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2, training=training)

        # output_seq2 = tf.transpose(output_seq2, [0, 2, 1, 3, 4])
        # output_seq3, input_h, input_c = ClockworkLayer(output_seq2.shape, rnn_layer=self.encoder3,
        #                                                k=self.period[2]).call(output_seq2, training=True)

        cwrnn3 = ClockworkLayer(input_shape, rnn_layer=self.encoder3, k=self.period[2],
                                embedding_layer=self.conv_embedding3, state_shape=output_seq2.shape, clockwork=self.cw)
        output_seq3, input_h, input_c = cwrnn3.call(x, output_seq2, training=True)
        layer_outputs.append(output_seq3)
        layer_h.append(input_h)
        layer_c.append(input_c)

        if self.rnn_bn:
            output_seq3 = self.bn3(output_seq3, training=training)

        # output_seq3 = tf.transpose(output_seq3, [0, 2, 1, 3, 4])
        # output_seq4, input_h, input_c = ClockworkLayer(output_seq3.shape, rnn_layer=self.encoder4,
        #                                                k=self.period[3]).call(output_seq3, training=True)

        cwrnn4 = ClockworkLayer(input_shape, rnn_layer=self.encoder4, k=self.period[3],
                                embedding_layer=self.conv_embedding4, state_shape=output_seq3.shape, clockwork=self.cw)
        output_seq4, input_h, input_c = cwrnn4.call(x, output_seq3, training=True)
        layer_outputs.append(output_seq4)
        layer_h.append(input_h)
        layer_c.append(input_c)

        return layer_outputs, layer_h, layer_c

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)


class VideoDecoderConvLSTM(tf.keras.Model):
    '''
    Decoder for creating fake video sequence (y_1, y_2,...,y_T) from the latent variable Z
    and last hidden state from encoder.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''

    def __init__(self, batch_size, int_time_steps, pred_time_steps, state_size, x_width, x_height, z_width=5,
                 z_height=5, filter_size=64, bn=False, output_activation="sigmoid", nlstm=1,
                 cat=False, nchannel=3, dropout=0.0, reg=False, rnn_dropout=0.0, cw=False, period=[1, 2, 4]):
        super(VideoDecoderConvLSTM, self).__init__()
        self.batch_size = batch_size
        self.int_time_steps = int_time_steps
        self.pred_time_steps = pred_time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.nchannel = nchannel
        self.rnn_bn = bn
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.cw = cw
        self.period = np.asarray(sorted(period, reverse=True))
        self.factor1 = self.period[0] // self.period[1]
        self.factor2 = self.period[1] // self.period[2]
        self.factor3 = self.period[2] // self.period[3]
        self.factor4 = 1
        self.period_time_steps = self.pred_time_steps // self.period
        if reg:
            self.reg = tf.keras.regularizers.L2(0.01)
        else:
            self.reg = None

        # compute paddings
        if self.x_height == self.x_width:
            stride3 = [2, 2]
            k_size3 = [6, 6]

            stride2 = [2, 2]
            k_size2 = [4, 4]

            stride1 = [2, 2]
            k_size1 = [2, 2]
        elif self.x_height < self.x_width:
            stride3 = [2, 2]
            k_size3 = [6, 7]

            stride2 = [2, 2]
            k_size2 = [6, 7]

            stride1 = [2, 2]
            k_size1 = [6, 7]
        else:
            stride4 = [3, 2]
            k_size4 = [9, 8]

            stride3 = [3, 2]
            k_size3 = [7, 6]

            stride2 = [3, 2]
            k_size2 = [7, 6]

            stride1 = [3, 2]
            k_size1 = [7, 6]
        #
        # decoder
        #
        self.decoder1 = tf.keras.layers.ConvLSTM2D(self.filter_size*16, kernel_size=(5, 5), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout,
                                                   kernel_regularizer=self.reg, recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(self.filter_size * 16, k_size1, strides=stride1,
                                                               padding='same', use_bias=False,
                                                               activation=output_activation, data_format='channels_last',
                                                               kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg
                                                               )
        self.conv_bn1 = tf.keras.layers.BatchNormalization()

        self.decoder2 = tf.keras.layers.ConvLSTM2D(self.filter_size * 8, kernel_size=(9, 9), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(self.filter_size * 8, k_size2, strides=stride2, padding='same',
                                                               use_bias=False, activation=output_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

        self.conv_bn2 = tf.keras.layers.BatchNormalization()

        self.decoder3 = tf.keras.layers.ConvLSTM2D(self.filter_size * 4, kernel_size=(9, 9), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg)
        self.bn6 = tf.keras.layers.BatchNormalization()

        self.conv_transpose3 = tf.keras.layers.Conv2DTranspose(self.filter_size * 2, k_size3, strides=stride3, padding='same',
                                                               use_bias=False, activation=output_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

        self.conv_bn3 = tf.keras.layers.BatchNormalization()

        self.decoder4 = tf.keras.layers.ConvLSTM2D(self.filter_size * 2, kernel_size=(9, 9), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=True,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg)
        self.bn7 = tf.keras.layers.BatchNormalization()

        self.conv_transpose4 = tf.keras.layers.Conv2DTranspose(self.nchannel, k_size3, strides=stride3, padding='same',
                                                               use_bias=False, activation=output_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

    def computing_padding(self, h_out, h_in, k_size, stride):
        p1 = ((h_in[0] - 1) * stride[0] - h_out[0] + k_size[0]) / 2
        p2 = ((h_in[1] - 1) * stride[1] - h_out[1] + k_size[1]) / 2
        padding_h = int(abs(tf.math.ceil(p1)))
        padding_w = int(abs(tf.math.ceil(p2)))
        return [padding_h, padding_w]

    def call_all(self, inputs_z, input_h, input_c, training=True):
    #def call_all(self, inputs_z, training=True):
        # z has shape of [batch_size, time_step, sub_sequence_hidden_dims]

        z = tf.reshape(tensor=inputs_z, shape=[self.batch_size, self.period_time_steps[0],
                                               self.z_height, self.z_width, -1])
        # input_h, input_c = [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 7]
        # h_dims = input_h.shape[-1]
        # h = tf.broadcast_to(tf.expand_dims(input_h, 1), shape=[self.batch_size, self.pred_time_steps, self.z_height,
        #                                                        self.z_width, h_dims])
        # z_h = tf.concat((z, h), axis=-1)
        # output_seq4, h, c = self.decoder1([z_h, input_h, input_c], training=training)

        # first period/lag, e.g., k=1 for no skipping
        cwrnn1 = ClockworkLayer(z.shape, self.decoder1, k=self.period[0], decoder=True, factor=self.factor1,
                                clockwork=self.cw)
        output_seq4, h, c = cwrnn1.call([z, input_h[3], input_c[3]], training=True)
        if self.rnn_bn:
            output_seq4 = self.bn4(output_seq4, training=training)

        # if self.x_height == self.x_width:
        #     padded_seq4 = tf.keras.layers.ZeroPadding3D(padding=(0, 4, 4))(output_seq4)
        # else:
        #     padded_seq4 = tf.keras.layers.ZeroPadding3D(padding=(0, 0, 0))(output_seq4)

        # output_seq5, h, c = self.decoder2(padded_seq4, training=training)

        conv_inputs1 = tf.reshape(output_seq4, [self.batch_size * self.period_time_steps[1], output_seq4.shape[2],
                                                output_seq4.shape[3], -1])

        conv_t = self.conv_transpose1(conv_inputs1)
        if self.rnn_bn:
            conv_t = self.conv_bn1(conv_t, training=training)

        lstm_inputs1 = tf.reshape(conv_t, [self.batch_size, self.period_time_steps[1], conv_t.shape[1],
                                           conv_t.shape[2], -1])

        cwrnn2 = ClockworkLayer(lstm_inputs1.shape, self.decoder2, k=self.period[1],
                                decoder=True, factor=self.factor2, clockwork=self.cw)
        output_seq5, h, c = cwrnn2.call([lstm_inputs1, input_h[2], input_c[2]], training=True)

        if self.rnn_bn:
            output_seq5 = self.bn5(output_seq5, training=training)

        conv_inputs2 = tf.reshape(output_seq5, [self.batch_size * self.period_time_steps[2], output_seq5.shape[2],
                                                output_seq5.shape[3], -1])

        conv_t = self.conv_transpose2(conv_inputs2)
        if self.rnn_bn:
            conv_t = self.conv_bn2(conv_t, training=training)

        lstm_inputs2 = tf.reshape(conv_t, [self.batch_size, self.period_time_steps[2], conv_t.shape[1],
                                           conv_t.shape[2], -1])

        cwrnn3 = ClockworkLayer(lstm_inputs2.shape, self.decoder3, k=self.period[2],
                                decoder=True, factor=self.factor3, clockwork=self.cw)
        output_seq6, h, c = cwrnn3.call([lstm_inputs2, input_h[1], input_c[1]], training=True)

        if self.rnn_bn:
            output_seq6 = self.bn6(output_seq6, training=training)

        conv_inputs3 = tf.reshape(output_seq6, [self.batch_size * self.period_time_steps[3], output_seq6.shape[2],
                                                output_seq6.shape[3], -1])

        conv_t = self.conv_transpose3(conv_inputs3)
        if self.rnn_bn:
            conv_t = self.conv_bn3(conv_t, training=training)

        lstm_inputs3 = tf.reshape(conv_t, [self.batch_size, self.period_time_steps[3], conv_t.shape[1],
                                           conv_t.shape[2], -1])

        cwrnn4 = ClockworkLayer(lstm_inputs3.shape, self.decoder4, k=self.period[3],
                                decoder=True, factor=self.factor4, clockwork=self.cw)
        output_seq7, h, c = cwrnn4.call([lstm_inputs3, input_h[0], input_c[0]], training=True)

        if self.rnn_bn:
            output_seq7 = self.bn7(output_seq7, training=training)

        conv_inputs4 = tf.reshape(output_seq7, [self.batch_size * self.pred_time_steps, output_seq7.shape[2],
                                                output_seq7.shape[3], -1])

        y = self.conv_transpose4(conv_inputs4)

        y = tf.reshape(y, [self.batch_size, self.pred_time_steps, self.x_height, self.x_width, self.nchannel])
        y = tf.transpose(y, (0, 2, 1, 3, 4))
        return y

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)


class VideoDiscriminator(tf.keras.Model):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, state_size, x_width, x_height, z_width=5, z_height=5,
                 filter_size=64, bn=False, output_activation="sigmoid", nlstm=1, cat=False, nchannel=3):
        super(VideoDiscriminator, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.bn = bn
        self.nchannel = nchannel

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(self.filter_size, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[x_height, x_width, nchannel]))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.filter_size * 2, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.filter_size * 4, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        self.conv = model

        self.rnn = tf.keras.Sequential()
        self.rnn.add(tf.keras.layers.LSTM(self.filter_size * 4, return_sequences=True))
        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.filter_size * 2, return_sequences=True))
        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):
        # permute original data shape [batch_size, h, timesteps, w, channels]
        # to [batch_size, timesteps, h, w, channels] as convnet inputs
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.x_height, self.time_steps,
                                             self.x_width, self.nchannel])
        z = tf.transpose(z, (0, 2, 1, 3, 4))
        z = tf.reshape(tensor=z, shape=[self.batch_size * self.time_steps, self.x_height, self.x_width, self.nchannel])

        z = self.conv(z)
        z = tf.reshape(z, shape=[self.batch_size, self.time_steps, -1])
        z = self.rnn(z)
        return z


class ClockworkLayer(tf.keras.layers.Layer):
    '''
    A custom layer which wraps RNN layers to create clockwork layers.

    '''

    def __init__(self, input_shape, rnn_layer, k, embedding_layer=None, state_shape=None,
                 decoder=False, factor=2, clockwork=False):
        super(ClockworkLayer, self).__init__()
        self.rnn_layer = rnn_layer
        self.embedding_layer = embedding_layer
        self.k = k
        self.decoder = decoder
        self.factor = factor
        self.clockwork = clockwork
        if clockwork:
            if not self.decoder:
                self.batch_size = input_shape[0]
                self.height = input_shape[1]
                self.time_steps = input_shape[2]
                self.width = input_shape[3]
                self.channels = input_shape[-1]
                self.n_states = self.time_steps // self.k

                self.mask = np.zeros([self.n_states, self.time_steps], np.float64)
                for t in range(self.n_states):
                    self.mask[t, t * self.k:(t + 1) * self.k] = 1.0

            if state_shape is not None:
                self.s_time_steps = state_shape[1]
                self.s_height = state_shape[2]
                self.s_width = state_shape[3]
                self.s_channels = state_shape[-1]

                self.state_mask = np.zeros([self.n_states, self.s_time_steps], np.float64)
                self.s_skip = self.s_time_steps // self.n_states
                for t in range(self.n_states):
                    self.state_mask[t, t * self.s_skip:(t + 1) * self.s_skip] = 1.0

                assert (self.height % self.s_height == 0 and self.width % self.s_width == 0), \
                    "Input size has to be twice state size."

    def call(self, inputs, states=None, **kwargs):
        if self.clockwork:
            if self.decoder:
                seq, h, c = inputs
                embedding = tf.repeat(seq, [self.factor], axis=1)
                # return self.rnn_layer([embedding, h, c])
                return self.rnn_layer(embedding)

            else:
                if self.k == 1 and self.embedding_layer is None:
                    embedding = tf.transpose(inputs, (0, 2, 1, 3, 4))
                    embedding = tf.reshape(embedding, [self.batch_size, self.n_states, self.height, self.width, -1])
                else:
                    # if self.k == 1 and states is not None:
                    #    embedding = states
                    # else:
                    broad_inputs = tf.broadcast_to(inputs,
                                                   (self.n_states, self.batch_size, self.height, self.time_steps,
                                                    self.width, self.channels))

                    masked_input = broad_inputs * self.mask[:, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
                    masked_input = tf.reduce_sum(masked_input, axis=3)
                    masked_input = tf.transpose(masked_input, (1, 0, 2, 3, 4))
                    masked_input = tf.reshape(masked_input,
                                              [self.n_states * self.batch_size, self.height, self.width, -1])

                    if states is not None:
                        states = tf.broadcast_to(states,
                                                 (self.n_states, self.batch_size, self.s_time_steps, self.s_height,
                                                  self.s_width, self.s_channels))
                        masked_states = states * self.state_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
                        masked_states = tf.reduce_sum(masked_states, axis=2)
                        masked_states = tf.transpose(masked_states, (1, 0, 2, 3, 4))

                        masked_states = tf.reshape(masked_states,
                                                   [self.batch_size, self.n_states, self.s_height, self.s_width, -1])
                        if self.embedding_layer is not None:
                            input_embedding = self.embedding_layer(masked_input)
                            input_embedding = tf.reshape(input_embedding,
                                                         [self.batch_size, self.n_states, self.s_height,
                                                          self.s_width, -1])
                            # embedding = tf.concat((input_embedding, masked_states), axis=-1)
                            embedding = input_embedding + masked_states
                        else:
                            embedding = masked_states
                    else:
                        if self.embedding_layer is not None:
                            embedding = self.embedding_layer(masked_input)
                            embedding = tf.reshape(embedding,
                                                   [self.batch_size, self.n_states, self.height, self.width, -1])
                        else:
                            embedding = tf.reshape(masked_input,
                                                   [self.n_states, self.batch_size, self.height, self.width,
                                                    -1])
                            embedding = tf.transpose(embedding, (1, 0, 2, 3, 4))
                return self.rnn_layer(embedding)
        else:
            if self.decoder:
                seq, h, c = inputs
                #embedding = tf.repeat(seq, [self.factor], axis=1)
                return self.rnn_layer([seq, h, c])
                # return self.rnn_layer(seq)
            else:
                inputs = tf.transpose(inputs, (0, 2, 1, 3, 4))
                if self.embedding_layer is not None:
                    inputs = self.embedding_layer(inputs)
                return self.rnn_layer(inputs)


class RNNClockworkLayer(tf.keras.layers.Layer):
    '''
    A custom layer which wraps RNN layers to create clockwork layers.

    '''

    def __init__(self, input_shape, rnn_layer, k, embedding_layer=None, state_shape=None,
                 decoder=False, factor=2):
        super(RNNClockworkLayer, self).__init__()
        self.rnn_layer = rnn_layer
        self.embedding_layer = embedding_layer
        self.k = k
        self.decoder = decoder
        self.factor = factor
        if not self.decoder:
            self.batch_size = input_shape[0]
            self.time_steps = input_shape[1]
            self.channels = input_shape[-1]
            self.n_states = self.time_steps // self.k

            self.mask = np.zeros([self.n_states, self.time_steps], np.float64)
            for t in range(self.n_states):
                self.mask[t, t * self.k:(t + 1) * self.k] = 1.0

        if state_shape is not None:
            self.s_time_steps = state_shape[1]
            self.s_channels = state_shape[-1]

            self.state_mask = np.zeros([self.n_states, self.s_time_steps], np.float64)
            self.s_skip = self.s_time_steps // self.n_states
            for t in range(self.n_states):
                self.state_mask[t, t * self.s_skip:(t + 1) * self.s_skip] = 1.0

    def call(self, inputs, states=None, **kwargs):
        if self.decoder:
            seq, h, c = inputs
            embedding = tf.repeat(seq, [self.factor], axis=1)
            return self.rnn_layer([embedding, h, c])

        else:
            if self.k == 1:
                embedding = tf.reshape(inputs, [self.batch_size, self.n_states, -1])
            else:
                broad_inputs = tf.broadcast_to(inputs, (self.n_states, self.batch_size, self.time_steps, self.channels))
                masked_input = broad_inputs * self.mask[:, tf.newaxis, :, tf.newaxis]
                masked_input = tf.reduce_sum(masked_input, axis=2)
                embedding = tf.transpose(masked_input, (1, 0, 2))

                if states is not None:
                    states = tf.broadcast_to(states, (self.n_states, self.batch_size, self.s_time_steps, self.s_channels))
                    masked_states = states * self.state_mask[:, tf.newaxis, :, tf.newaxis]
                    masked_states = tf.reduce_sum(masked_states, axis=2)
                    masked_states = tf.transpose(masked_states, (1, 0, 2))
                    embedding = tf.concat((embedding, masked_states), axis=-1)
            return self.rnn_layer(embedding)