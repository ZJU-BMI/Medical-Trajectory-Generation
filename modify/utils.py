import tensorflow as tf
import numpy as np
from tensorflow_core.python.keras.models import Model
import sys
import os


# transform each step of x, i.e. x_i into h_i
class Encoder(Model):
    def __init__(self, hidden_size):
        super().__init__(name='encode_share')
        self.hidden_size = hidden_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_x):
        sequence_time, c, h = input_x
        state = [c, h]
        output, state = self.LSTM_Cell_encode(sequence_time, state)
        return state[0], state[1]


# decode or generate the next sequence
class Decoder(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_share')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.LSTM_Cell_decode = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, input_x):
        sequence_time, encode_h, decode_c, decode_h = input_x
        decode_input = tf.concat((sequence_time, encode_h), axis=1)
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decode(decode_input, state)
        y_i = self.dense1(output)
        y_i = self.dense2(y_i)
        y_i = self.dense3(y_i)
        return y_i, state[0], state[1]


class HawkesProcess(Model):
    def __init__(self):
        super().__init__(name='point_process')

    def build(self, input_shape):
        shape_weight = tf.TensorShape((1, 1))
        self.trigger_parameter_alpha = self.add_weight(name='trigger_alpha',
                                                       shape=shape_weight,
                                                       initializer='uniform',
                                                       trainable=True)

        self.trigger_parameter_beta = self.add_weight(name='trigger_beta',
                                                      shape=shape_weight,
                                                      initializer='uniform',
                                                      trainable=True)

        self.base_intensity = self.add_weight(name='trigger_beta',
                                              shape=shape_weight,
                                              initializer='uniform',
                                              trainable=True)
        super(HawkesProcess, self).build(input_shape)

    def calculate_lambda_process(self, input_t, current_time_index, trigger_alpha, trigger_beta, base_intensity):
        batch = tf.shape(input_t)[0]
        current_t = tf.reshape(input_t[:, current_time_index], [batch, -1])
        current_t_tile = tf.tile(current_t, [1, current_time_index])

        time_before_t = input_t[:, :current_time_index]

        time_difference = time_before_t - current_t_tile

        trigger_kernel = tf.reduce_sum(tf.exp(trigger_beta * time_difference), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity_value = base_intensity + trigger_kernel * trigger_alpha
        return condition_intensity_value

    def calculate_likelihood(self, input_t, current_time_index, trigger_alpha, trigger_beta, base_intensity):
        batch = tf.shape(input_t)[0]
        ratio_alpha_beta = trigger_alpha / trigger_beta

        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])
        current_t_tile = tf.tile(current_t, [1, current_time_index])

        time_before_t = input_t[:, :current_time_index]

        # part_1: t_i -t(<i)
        time_difference = time_before_t - current_t_tile

        trigger_kernel = tf.reduce_sum(tf.exp(trigger_beta * time_difference), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        conditional_intensity = base_intensity + trigger_alpha * trigger_kernel  # part 1 result

        # part_2: t_i - t_(i-1)
        last_time = input_t[:, current_time_index-1]
        time_difference_2 = (tf.reshape(last_time, [batch, 1]) - current_t) * base_intensity  # part 2 result

        # part_3: t_(i-1) - t(<i)
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = time_before_t - last_time_tile
        time_difference_3 = tf.reduce_sum(tf.exp(time_difference_3 * trigger_beta), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, 1])

        probability_result = conditional_intensity * tf.exp(time_difference_2 + ratio_alpha_beta*(trigger_kernel - time_difference_3))

        return probability_result

    def call(self, input_x):
        input_t, current_time_index_shape = input_x
        current_time_index = tf.shape(current_time_index_shape)[0]
        batch = tf.shape(input_t)[0]
        trigger_alpha = tf.tile(self.trigger_parameter_alpha, [batch, 1])
        trigger_beta = tf.tile(self.trigger_parameter_beta, [batch, 1])
        base_intensity = tf.tile(self.base_intensity, [batch, 1])

        condition_intensity = self.calculate_lambda_process(input_t, current_time_index,
                                                            trigger_alpha, trigger_beta, base_intensity)
        likelihood = self.calculate_likelihood(input_t, current_time_index, trigger_alpha,
                                               trigger_beta, base_intensity)
        return condition_intensity, likelihood


def test_test(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(name)

    print(path)
    print(os.path.dirname(__file__))
    print('------------------')


if __name__ == '__main__':
    x = np.zeros(shape=(5, 2, 5))
    y = np.ones(shape=(5, 3, 5))
    m = np.concatenate((x, y), axis=1)

    n = np.maximum(m, 0.1)
    print(n)