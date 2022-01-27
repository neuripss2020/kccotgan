import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx('float32')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


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
        self.in_activation = 'tanh'
        self.norm_axis = [-1]
        #
        # ConvLSTM encoder
        #
        self.encoder1 = tf.keras.layers.ConvLSTM2D(self.filter_size * 4, kernel_size=(6, 6), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn1 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.encoder2 = tf.keras.layers.ConvLSTM2D(self.filter_size * 8, kernel_size=(6, 6), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn2 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.encoder3 = tf.keras.layers.ConvLSTM2D(self.filter_size * 16, kernel_size=(5, 5), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)

        self.bn3 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.encoder4 = tf.keras.layers.ConvLSTM2D(self.filter_size * 32, kernel_size=(5, 5), strides=(2, 2),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn4 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

    def call_all(self, inputs_real, training=True):
        x = tf.transpose(inputs_real, (0, 2, 1, 3, 4))
        pred_features = [x[:, self.int_time_steps-1:, ...]]

        output_seq1 = self.encoder1(x, training=training)
        if self.rnn_bn:
            output_seq1 = self.bn1(output_seq1, training=training)
        pred_features.append(output_seq1[:, self.int_time_steps-1:, ...])

        output_seq2 = self.encoder2(output_seq1, training=training)
        if self.rnn_bn:
            output_seq2 = self.bn2(output_seq2, training=training)
        pred_features.append(output_seq2[:, self.int_time_steps-1:, ...])

        output_seq3 = self.encoder3(output_seq2, training=training)
        if self.rnn_bn:
            output_seq3 = self.bn3(output_seq3, training=training)
        pred_features.append(output_seq3[:, self.int_time_steps-1:, ...])

        output_seq4 = self.encoder4(output_seq3, training=training)
        if self.rnn_bn:
            output_seq4 = self.bn4(output_seq4, training=training)
        pred_features.append(output_seq4[:, self.int_time_steps-1:, ...])
        return pred_features

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
            self.reg = tf.keras.regularizers.L2(0.1)
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

        self.in_activation = 'tanh'
        self.norm_axis = [-1]

        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(self.filter_size * 32, k_size1, strides=stride1,
                                                               padding='same', use_bias=False,
                                                               activation=self.in_activation, data_format='channels_last',
                                                               kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg
                                                               )

        self.conv_bn1 = tf.keras.layers.LayerNormalization(axis=[-1])

        self.decoder2 = tf.keras.layers.ConvLSTM2D(self.filter_size * 16, kernel_size=(4, 4), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg, bias_regularizer=self.reg,
                                                   activity_regularizer=self.reg)
        self.bn5 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(self.filter_size * 16, k_size2, strides=stride2, padding='same',
                                                               use_bias=False,  activation=self.in_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

        self.conv_bn2 = tf.keras.layers.LayerNormalization(axis=[-1])

        self.decoder3 = tf.keras.layers.ConvLSTM2D(self.filter_size * 8, kernel_size=(6, 6), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   use_bias=False, activation=self.in_activation,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg)
        self.bn6 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.conv_transpose3 = tf.keras.layers.Conv2DTranspose(self.filter_size*8, k_size3, strides=stride3,
                                                               padding='same', use_bias=False,
                                                               activation=self.in_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

        self.conv_bn3 = tf.keras.layers.LayerNormalization(axis=[-1])

        self.decoder4 = tf.keras.layers.ConvLSTM2D(self.filter_size * 4, kernel_size=(8, 8), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg,
                                                   activation=self.in_activation)
        self.bn7 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.conv_transpose4 = tf.keras.layers.Conv2DTranspose(self.filter_size*2, k_size3, strides=stride3,
                                                               padding='same',
                                                               use_bias=False, activation=self.in_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

        self.conv_bn4 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.decoder5 = tf.keras.layers.ConvLSTM2D(self.filter_size, kernel_size=(8, 8), strides=(1, 1),
                                                   padding='same', data_format='channels_last', return_state=False,
                                                   return_sequences=True, dropout=self.dropout,
                                                   recurrent_dropout=self.rnn_dropout, kernel_regularizer=self.reg,
                                                   recurrent_regularizer=self.reg,
                                                   bias_regularizer=self.reg, activity_regularizer=self.reg,
                                                   activation=self.in_activation)
        self.bn8 = tf.keras.layers.LayerNormalization(axis=self.norm_axis)

        self.conv_transpose5 = tf.keras.layers.Conv2DTranspose(self.nchannel, [8, 8], strides=[1, 1],
                                                               padding='same',
                                                               use_bias=False, activation=output_activation,
                                                               data_format='channels_last', kernel_regularizer=self.reg,
                                                               bias_regularizer=self.reg, activity_regularizer=self.reg)

    def call_all(self, predictions, inputs_z, training=True):
        if training:
            inp1 = predictions[4][:, :-1, :, :, :]
        else:
            inp1 = predictions[4][:, -1, :, :, :][:, tf.newaxis, :, :, :]
        inp1 = tf.concat((inp1, inputs_z), axis=-1)
        time = inputs_z.shape[1]

        conv_inputs1 = tf.reshape(inp1, [self.batch_size * time, inp1.shape[2], inp1.shape[3], -1])

        conv_t = self.conv_transpose1(conv_inputs1)
        # conv_t = self.act1(conv_t)
        if self.rnn_bn:
            conv_t = self.conv_bn1(conv_t, training=training)

        lstm_inputs1 = tf.reshape(conv_t, [self.batch_size, time, conv_t.shape[1],
                                           conv_t.shape[2], -1])

        if training:
            inp2 = predictions[3][:, :-1, :, :, :]
        else:
            inp2 = predictions[3][:, -1, :, :, :][:, tf.newaxis, :, :, :]
        inp2 = tf.concat((inp2, lstm_inputs1), axis=-1)

        output_seq5 = self.decoder2(inp2, training=training)

        if self.rnn_bn:
            output_seq5 = self.bn5(output_seq5, training=training)

        conv_inputs2 = tf.reshape(output_seq5, [self.batch_size * time, output_seq5.shape[2],
                                                output_seq5.shape[3], -1])

        conv_t = self.conv_transpose2(conv_inputs2)
        if self.rnn_bn:
            conv_t = self.conv_bn2(conv_t, training=training)

        lstm_inputs2 = tf.reshape(conv_t, [self.batch_size, time, conv_t.shape[1],
                                           conv_t.shape[2], -1])
        if training:
            inp3 = predictions[2][:, :-1, :, :, :]
        else:
            inp3 = predictions[2][:, -1, :, :, :][:, tf.newaxis, :, :, :]
        inp3 = tf.concat((inp3, lstm_inputs2), axis=-1)

        output_seq6 = self.decoder3(inp3, training=training)

        if self.rnn_bn:
            output_seq6 = self.bn6(output_seq6, training=training)

        conv_inputs3 = tf.reshape(output_seq6, [self.batch_size * time, output_seq6.shape[2],
                                                output_seq6.shape[3], -1])

        conv_t = self.conv_transpose3(conv_inputs3)
        if self.rnn_bn:
            conv_t = self.conv_bn3(conv_t, training=training)

        lstm_inputs3 = tf.reshape(conv_t, [self.batch_size, time, conv_t.shape[1],
                                           conv_t.shape[2], -1])
        if training:
            inp4 = predictions[1][:, :-1, :, :, :]
        else:
            inp4 = predictions[1][:, -1, :, :, :][:, tf.newaxis, :, :, :]
        inp4 = tf.concat((inp4, lstm_inputs3), axis=-1)

        output_seq7 = self.decoder4(inp4, training=training)
        if self.rnn_bn:
            output_seq7 = self.bn7(output_seq7, training=training)

        conv_inputs4 = tf.reshape(output_seq7, [self.batch_size * time, output_seq7.shape[2],
                                                output_seq7.shape[3], -1])

        conv_t = self.conv_transpose4(conv_inputs4)
        if self.rnn_bn:
            conv_t = self.conv_bn4(conv_t, training=training)

        lstm_inputs4 = tf.reshape(conv_t, [self.batch_size, time, conv_t.shape[1],
                                           conv_t.shape[2], -1])
        if training:
            inp5 = predictions[0][:, :-1, :, :, :]
        else:
            inp5 = predictions[0][:, -1, :, :, :][:, tf.newaxis, :, :, :]
        inp5 = tf.concat((inp5, lstm_inputs4), axis=-1)

        output_seq8 = self.decoder5(inp5, training=training)
        if self.rnn_bn:
            output_seq8 = self.bn8(output_seq8, training=training)

        conv_inputs5 = tf.reshape(output_seq8, [self.batch_size * time, output_seq8.shape[2],
                                                output_seq8.shape[3], -1])

        conv_t = self.conv_transpose5(conv_inputs5)
        y = tf.reshape(conv_t, [self.batch_size, time, self.x_height, self.x_width, self.nchannel])
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
        model.add(tf.keras.layers.Conv2D(self.filter_size*4, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[x_height, x_width, nchannel]))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.filter_size * 8, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.filter_size * 16, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        self.conv = model

        self.rnn = tf.keras.Sequential()
        self.rnn.add(tf.keras.layers.LSTM(self.filter_size * 8, return_sequences=True))
        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.filter_size * 4, return_sequences=True))
        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.state_size, return_sequences=True, activation=output_activation))

    def call(self, inputs, training=True, mask=None):
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.x_height, self.time_steps,
                                             self.x_width, self.nchannel])
        z = tf.transpose(z, (0, 2, 1, 3, 4))
        z = tf.reshape(tensor=z, shape=[self.batch_size * self.time_steps, self.x_height, self.x_width, self.nchannel])

        z = self.conv(z)
        z = tf.reshape(z, shape=[self.batch_size, self.time_steps, -1])
        z = self.rnn(z)
        return z