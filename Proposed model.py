import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet
from LSTMCell import *
from bayes_opt import BayesianOptimization
from scipy import stats
import os
import sys
from sklearn.preprocessing import MinMaxScaler


class ShareInformationEncode(tf.keras.layers.Layer):
    def __init__(self, encode_dims):
        super(ShareInformationEncode, self).__init__()
        self.encode_dims = encode_dims
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(encode_dims)

    def call(self, state_encode, real_sequence_time):  # [c, h], input_x_next
        output, output_state = self.LSTM_Cell_encode(real_sequence_time, state_encode)
        return output, output_state


# encode the information from time t_1 to time t_i into hidden representation,
# and return the last time hidden representation into next process
class EncoderHistoryInformation(Model):
    def __init__(self, time_step, hidden_size, batch_size, previous_visit):
        super().__init__(name='encode_context')
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.previous_visit = previous_visit
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)
        # self.share_encode = ShareInformationEncode(encode_dims=hidden_size)

    def call(self, input_context):
        batch = tf.shape(input_context)[0]
        state_forward = self.LSTM_Cell_encode.get_initial_state(batch_size=batch, dtype=tf.float32)
        output_forward = None
        for time in range(self.previous_visit):
            input_time_forward = input_context[:, time, :]
            output_forward, state_forward = self.LSTM_Cell_encode(input_time_forward, state_forward)
        return output_forward, state_forward   # [batch, hidden_size]


class VAE(Model):
    def __init__(self, z_dims, batch_size, encode_dims, predicted_visit, feature_dims, previous_visit):
        super().__init__(name='VAE')
        self.z_dims = z_dims
        self.predicted_visit = predicted_visit
        self.batch_size = batch_size
        self.encode_dims = encode_dims
        self.feature_dims = feature_dims
        self.previous_visit = previous_visit
        self.LSTM_Cell_forward = tf.keras.layers.LSTMCell(encode_dims)
        # obtain z
        self.dense1 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        # obtain z_mean and z_log_var
        self.dense4 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        # reconstruct input from sample z
        self.share_encode = tf.keras.layers.LSTMCell(encode_dims)
        self.share_decode = tf.keras.layers.LSTMCell(feature_dims)
        # obtain next visit
        self.dense6 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense7 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense8 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, inputs_vae):
        h_i, h_i_information, real_sequence, input_t = inputs_vae
        c = h_i_information[:, :self.encode_dims]
        h = h_i_information[:, self.encode_dims:]
        state_encode = [c, h]
        batch = tf.shape(real_sequence)[0]
        z_all = tf.zeros(shape=[batch, 0, self.z_dims])
        z_mean_all = tf.zeros(shape=(batch, 0, self.z_dims))
        z_log_var_all = tf.zeros(shape=(batch, 0, self.z_dims))
        input_reconstruction = tf.zeros(shape=(batch, 0, self.feature_dims))
        state_decode = self.share_decode.get_initial_state(batch_size=batch, dtype=tf.float32)
        for time in range(self.predicted_visit):
            sequence_time = real_sequence[:, self.previous_visit + time, :]
            h_i_ = h_i  # store the last visit hidden representation
            output, state_encode = self.share_encode(sequence_time, state_encode)  # h_(i+1)
            hidden_concat = tf.concat((h_i, output), axis=1)  # [batch, encode_dims*2] concat hidden state
            h_i = output  # update
            hidden_concat_1 = self.dense1(hidden_concat)
            hidden_concat_2 = self.dense2(hidden_concat_1)
            hidden_concat_3 = self.dense3(hidden_concat_2)  # hidden_z
            mean_z = self.dense4(hidden_concat_3)
            log_var_z = self.dense5(hidden_concat_3)
            log_var_z = tf.nn.softplus(log_var_z) + 1e-6
            z = self.reparameterize(mean_z, log_var_z)  # z_(i+1)
            input_decode = tf.concat((z, h_i_), axis=1)  # z_(i+1) and h_i
            batch = tf.shape(input_t)[0]
            current_time_index = time + self.predicted_visit
            condition_value = self.calculate_hawkes_process(batch=batch, input_t=input_t,
                                                            current_time_index=current_time_index,
                                                            trigger_parameter_alpha=self.trigger_parameter_alpha,
                                                            trigger_parameter_beta=self.trigger_parameter_beta,
                                                            base_intensity=self.base_intensity)
            state_decode = [state_decode[0] * condition_value, state_decode[1] * condition_value]
            # recurrence and generate next visit hidden representation
            generated_next_visit_hidden, state_decode = self.share_decode(input_decode, state_decode)

            generated_next_visit = self.dense6(generated_next_visit_hidden)
            generated_next_visit = self.dense7(generated_next_visit)
            generated_next_visit = self.dense8(generated_next_visit)

            z_mean_all = tf.concat((z_mean_all, tf.reshape(mean_z, [batch, -1, self.z_dims])), axis=1)
            z_log_var_all = tf.concat((z_log_var_all, tf.reshape(log_var_z, [batch, -1, self.z_dims])), axis=1)
            z_all = tf.concat((z_all, tf.reshape(z, [batch, -1, self.z_dims])), axis=1)

            generated_next_visit = tf.reshape(generated_next_visit, [batch, -1, self.feature_dims])
            input_reconstruction = tf.concat((input_reconstruction, generated_next_visit), axis=1)

        return z_all, z_mean_all, z_log_var_all, input_reconstruction

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.compat.v1.random_normal(shape=tf.shape(std))
        return mu + eps * std

    def build(self, input_shape):
        shape_hawkes = tf.TensorShape((1, 1))
        self.trigger_parameter_alpha = self.add_weight(name='trigger_alpha',
                                                       shape=shape_hawkes,
                                                       initializer='uniform',
                                                       trainable=True)
        self.trigger_parameter_beta = self.add_weight(name='trigger_beta',
                                                      shape=shape_hawkes,
                                                      initializer='uniform',
                                                      trainable=True)
        self.base_intensity = self.add_weight(name='base_intensity',
                                              shape=shape_hawkes,
                                              initializer='uniform',
                                              trainable=True)

        super(VAE, self).build(input_shape)

        # intensity function: lambda(t))
    def calculate_hawkes_process(self, batch, input_t, current_time_index,
                                 trigger_parameter_beta, trigger_parameter_alpha, base_intensity):
        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])  # number value[batch ,1]
        current_t_tile = tf.tile(current_t, [1, current_time_index])  # [batch, current_index]

        time_before_t = input_t[:, :current_time_index]
        time_difference = time_before_t - current_t_tile  # [batch, current_index]

        triggering_kernel = tf.reduce_sum(tf.exp(time_difference * trigger_parameter_beta), axis=1)

        triggering_kernel = tf.reshape(triggering_kernel, [batch, 1])
        condition_intensity_value = base_intensity + trigger_parameter_alpha * triggering_kernel
        return condition_intensity_value

        # （1）按照仅仅和时间有关的公式推导 已知1...j 推测j+1次

    def calculate_likelihood(self, batch, input_t, current_time_index, trigger_parameter_beta,
                             trigger_parameter_alpha, base_intensity):
        ratio_alpha_beta = trigger_parameter_alpha / trigger_parameter_beta
        # the current time
        current_time = input_t[:, current_time_index]
        current_time = tf.reshape(current_time, [batch, -1])

        time_before_t = input_t[:, :current_time_index]

        current_time_tile = tf.tile(current_time, [1, current_time_index])  # [batch, current_time_index]

        # part_1: t_i - t_3
        time_difference_1 = time_before_t - current_time_tile
        trigger_kernel = tf.reduce_sum(tf.exp(trigger_parameter_beta * time_difference_1), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity = base_intensity + trigger_parameter_alpha * trigger_kernel  # [batch, 1]

        # part_2: t_2 - t_3
        last_time = tf.reshape(input_t[:, current_time_index - 1], [batch, 1])
        time_difference_2 = (last_time - current_time) * base_intensity  # [batch, 1]

        # part_3: t_i -t_2
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = tf.reduce_sum(tf.exp(trigger_parameter_beta * (time_before_t - last_time_tile)), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, -1])

        probability_result = condition_intensity * tf.exp(
            time_difference_2 + ratio_alpha_beta * (trigger_kernel - time_difference_3))

        probability_result = tf.reshape(probability_result, [batch, 1])
        return probability_result


class EncodeModify(Model):
    def __init__(self, hidden_size):
        super().__init__(name='encode_share_modify')
        self.hidden_size = hidden_size
        self.LSTM_Cell_forward = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, inputs_encode):
        input_x, encode_history = inputs_encode
        c = encode_history[:, :self.hidden_size]
        h = encode_history[:, self.hidden_size:]
        encode_history = [c, h]
        output, output_state = self.LSTM_Cell_forward(input_x, encode_history)
        output_state = tf.concat((output_state[0], output_state[1]), axis=1)
        return output, output_state


# z---->x and f(t)
class ShareInformationDecode(tf.keras.layers.Layer):
    def __init__(self, feature_dims, batch_size):
        self.feature_dims = feature_dims
        self.batch_size = batch_size
        self.LSTM_Cell_decode = tf.keras.layers.LSTMCell(feature_dims)
        self.trigger_parameter_alpha = None
        self.trigger_parameter_beta = None
        self.base_intensity = None

    def build(self, input_shape):
        shape_hawkes = tf.TensorShape((1, 1))
        self.trigger_parameter_alpha = self.add_weight(name='trigger_alpha',
                                                       shape=shape_hawkes,
                                                       initializer='uniform',
                                                       trainable=True)
        self.trigger_parameter_beta = self.add_weight(name='trigger_beta',
                                                      shape=shape_hawkes,
                                                      initializer='uniform',
                                                      trainable=True)
        self.base_intensity = self.add_weight(name='base_intensity',
                                              shape=shape_hawkes,
                                              initializer='uniform',
                                              trainable=True)

        super(ShareInformationDecode, self).build(input_shape)

    # intensity function: lambda(t))
    def calculate_hawkes_process(self, batch, input_t, current_time_index,
                                 trigger_parameter_beta, trigger_parameter_alpha, base_intensity):
        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])  # number value[batch ,1]
        current_t_tile = tf.tile(current_t, [1, current_time_index])  # [batch, current_index]

        time_before_t = input_t[:, :current_time_index]
        time_difference = time_before_t - current_t_tile  # [batch, current_index]

        triggering_kernel = tf.reduce_sum(tf.exp(time_difference * trigger_parameter_beta), axis=1)

        triggering_kernel = tf.reshape(triggering_kernel, [batch, 1])
        condition_intensity_value = base_intensity + trigger_parameter_alpha * triggering_kernel
        return condition_intensity_value

    # （1）按照仅仅和时间有关的公式推导 已知1...j 推测j+1次
    def calculate_likelihood(self, batch, input_t, current_time_index, trigger_parameter_beta,
                             trigger_parameter_alpha, base_intensity):
        ratio_alpha_beta = trigger_parameter_alpha / trigger_parameter_beta
        # the current time
        current_time = input_t[:, current_time_index]
        current_time = tf.reshape(current_time, [batch, -1])

        time_before_t = input_t[:, :current_time_index]

        current_time_tile = tf.tile(current_time, [1, current_time_index])  # [batch, current_time_index]

        # part_1: t_i - t_3
        time_difference_1 = time_before_t - current_time_tile
        trigger_kernel = tf.reduce_sum(tf.exp(trigger_parameter_beta * time_difference_1), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity = base_intensity + trigger_parameter_alpha * trigger_kernel  # [batch, 1]

        # part_2: t_2 - t_3
        last_time = tf.reshape(input_t[:, current_time_index - 1], [batch, 1])
        time_difference_2 = (last_time - current_time) * base_intensity  # [batch, 1]

        # part_3: t_i -t_2
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = tf.reduce_sum(tf.exp(trigger_parameter_beta * (time_before_t - last_time_tile)), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, -1])

        probability_result = condition_intensity * tf.exp(
            time_difference_2 + ratio_alpha_beta * (trigger_kernel - time_difference_3))

        probability_result = tf.reshape(probability_result, [batch, 1])
        return probability_result

    def call(self, input_x, state_decode, current_time_index, input_t):
        batch = tf.shape(input_t)[0]
        condition_value = self.calculate_hawkes_process(batch=batch, input_t=input_t, current_time_index=current_time_index-1,
                                                        trigger_parameter_alpha=self.trigger_parameter_alpha,
                                                        trigger_parameter_beta=self.trigger_parameter_beta,
                                                        base_intensity=self.base_intensity)

        likelihood = self.calculate_likelihood(batch=batch, input_t=input_t, current_time_index=current_time_index-1,
                                               trigger_parameter_alpha=self.trigger_parameter_alpha,
                                               trigger_parameter_beta=self.trigger_parameter_beta,
                                               base_intensity=self.base_intensity)
        state_decode = state_decode * condition_value
        output, output_state = self.LSTM_Cell_decode(input_x, state_decode)
        output = tf.nn.sigmoid(output)
        return output, likelihood


# 只能每次写一步 不能写多步 因为测试和训练的时候输入不一致
class PriorAndGenerationNetwork(Model):
    def __init__(self, z_dims, batch_size, encode_dims, predicted_visit, feature_dims, previous_visit):
        super().__init__(name='Prior_and_generate_network')
        self.z_dims = z_dims
        self.batch_size = batch_size
        self.encode_dims = encode_dims
        self.predicted_visit = predicted_visit
        self.feature_dims = feature_dims
        self.predicted_visit = previous_visit
        self.LSTM_Cell_forward = tf.keras.layers.LSTMCell(encode_dims)

        # obtain z
        self.dense1 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)

        # obtain z_mean and z_log_var
        self.dense4 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        # generate x_(i+1)
        self.share_decode = tf.keras.layers.LSTMCell(feature_dims)
        # obtain the next visit
        self.dense6 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense7 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense8 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def build(self, input_shape):
        shape_hawkes = tf.TensorShape((1, 1))
        self.trigger_parameter_alpha = self.add_weight(name='trigger_alpha',
                                                       shape=shape_hawkes,
                                                       initializer='uniform',
                                                       trainable=True)
        self.trigger_parameter_beta = self.add_weight(name='trigger_beta',
                                                      shape=shape_hawkes,
                                                      initializer='uniform',
                                                      trainable=True)
        self.base_intensity = self.add_weight(name='base_intensity',
                                              shape=shape_hawkes,
                                              initializer='uniform',
                                              trainable=True)

        super(PriorAndGenerationNetwork, self).build(input_shape)

        # intensity function: lambda(t))
    def calculate_hawkes_process(self, batch, input_t, current_time_index,
                                 trigger_parameter_beta, trigger_parameter_alpha, base_intensity):
        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])  # number value[batch ,1]
        current_t_tile = tf.tile(current_t, [1, current_time_index])  # [batch, current_index]

        time_before_t = input_t[:, :current_time_index]
        time_difference = time_before_t - current_t_tile  # [batch, current_index]

        triggering_kernel = tf.reduce_sum(tf.exp(time_difference * trigger_parameter_beta), axis=1)

        triggering_kernel = tf.reshape(triggering_kernel, [batch, 1])
        condition_intensity_value = base_intensity + trigger_parameter_alpha * triggering_kernel
        return condition_intensity_value

        # （1）按照仅仅和时间有关的公式推导 已知1...j 推测j+1次

    def calculate_likelihood(self, batch, input_t, current_time_index, trigger_parameter_beta,
                             trigger_parameter_alpha, base_intensity):
        ratio_alpha_beta = trigger_parameter_alpha / trigger_parameter_beta
        # the current time
        current_time = input_t[:, current_time_index]
        current_time = tf.reshape(current_time, [batch, -1])

        time_before_t = input_t[:, :current_time_index]

        current_time_tile = tf.tile(current_time, [1, current_time_index])  # [batch, current_time_index]

        # part_1: t_i - t_3
        time_difference_1 = time_before_t - current_time_tile
        trigger_kernel = tf.reduce_sum(tf.exp(trigger_parameter_beta * time_difference_1), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity = base_intensity + trigger_parameter_alpha * trigger_kernel  # [batch, 1]

        # part_2: t_2 - t_3
        last_time = tf.reshape(input_t[:, current_time_index - 1], [batch, 1])
        time_difference_2 = (last_time - current_time) * base_intensity  # [batch, 1]

        # part_3: t_i -t_2
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = tf.reduce_sum(tf.exp(trigger_parameter_beta * (time_before_t - last_time_tile)), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, -1])

        probability_result = condition_intensity * tf.exp(
            time_difference_2 + ratio_alpha_beta * (trigger_kernel - time_difference_3))

        probability_result = tf.reshape(probability_result, [batch, 1])
        return probability_result

    def prior_work(self, h_i):
        hidden_z_1 = self.dense1(h_i)
        hidden_z_2 = self.dense2(hidden_z_1)
        hidden_z_3 = self.dense3(hidden_z_2)  # z hidden representation
        z_mean = self.dense4(hidden_z_3)
        z_log_var = self.dense5(hidden_z_3)
        z_log_var = tf.nn.softplus(z_log_var) + 1e-6
        return z_mean, z_log_var

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.compat.v1.random_normal(shape=tf.shape(std))
        return mu + eps * std

    def call(self, inputs_prior):

        h_i, h_i_information, h_i_recurrence, h_i_state_recurrence, real_sequence_last_time, real_sequence_current_time, current_time_index_shape, input_t = inputs_prior

        c = h_i_state_recurrence[:, :self.feature_dims]
        h = h_i_state_recurrence[:, self.feature_dims:]
        state_encode = [c, h]

        z_mean, z_log_var = self.prior_work(h_i)
        z = self.reparameterize(z_mean, z_log_var)  # z_i
        input_decode = tf.concat((z, h_i), axis=1)  # concat z_(i+1) and h_i

        batch = tf.shape(input_t)[0]
        current_time_index = tf.shape(current_time_index_shape)[0]

        condition_value = self.calculate_hawkes_process(batch=batch, input_t=input_t,
                                                        current_time_index=current_time_index,
                                                        trigger_parameter_alpha=self.trigger_parameter_alpha,
                                                        trigger_parameter_beta=self.trigger_parameter_beta,
                                                        base_intensity=self.base_intensity)

        likelihood = self.calculate_likelihood(batch=batch, input_t=input_t, current_time_index=current_time_index,
                                               trigger_parameter_alpha=self.trigger_parameter_alpha,
                                               trigger_parameter_beta=self.trigger_parameter_beta,
                                               base_intensity=self.base_intensity)

        state_decode = [state_encode[0] * condition_value, state_encode[1]*condition_value]

        output, output_state = self.share_decode(input_decode, state_decode)

        generated_next_visit = self.dense6(output)
        generated_next_visit = self.dense7(generated_next_visit)
        generated_next_visit = self.dense8(generated_next_visit)
        generated_next_visit = tf.nn.sigmoid(generated_next_visit)

        output_state = tf.concat((output_state[0], output_state[1]), axis=1)
        return z, z_mean, z_log_var, generated_next_visit, likelihood, output, output_state


# discriminator network
class Discriminator(Model):
    def __init__(self, time_step, batch_size, hidden_size, previous_visit, predicted_visit):
        super().__init__(name='discriminator')
        self.time_step = time_step
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit

        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, real_input, fake_input):
        batch = tf.shape(real_input)[0]

        input_same = real_input[:, :self.previous_visit, :]
        input_same_real = input_same
        input_same_fake = input_same

        trajectory_real = []
        trajectory_fake = []

        trajectory_real_predict = tf.zeros(shape=[batch, 0, 1])
        trajectory_fake_predict = tf.zeros(shape=[batch, 0, 1])
        for index in range(self.predicted_visit):
            next_real = real_input[:, index+self.previous_visit, :]
            next_fake = fake_input[:, index, :]
            next_real = tf.reshape(next_real, [batch, 1, -1])
            next_fake = tf.reshape(next_fake, [batch, 1, -1])
            trajectory_step_real = tf.concat((input_same_real, next_real), axis=1)
            trajectory_step_fake = tf.concat((input_same_fake, next_fake), axis=1)

            trajectory_real.append(trajectory_step_real)
            trajectory_fake.append(trajectory_step_fake)

            input_same_real = trajectory_step_real
            input_same_fake = trajectory_step_fake

        for time_index in range(self.predicted_visit):
            output_real = None
            output_fake = None
            trajectory_real_ = trajectory_real[time_index]
            trajectory_fake_ = trajectory_fake[time_index]

            state = self.LSTM_Cell.get_initial_state(batch_size=batch, dtype=tf.float32)
            state_real = state
            state_fake = state
            for t in range(tf.shape(trajectory_real_)[1]):
                input_real = trajectory_real_[:, t, :]
                input_fake = trajectory_fake_[:, t, :]
                output_real, state_real = self.LSTM_Cell(input_real, state_real)
                output_fake, state_fake = self.LSTM_Cell(input_fake, state_fake)

            output_fake = self.dense1(output_fake)
            output_real = self.dense1(output_real)

            trajectory_step_real_pre = self.dense2(output_real)
            trajectory_step_fake_pre = self.dense2(output_fake)

            trajectory_step_real_pre = tf.reshape(trajectory_step_real_pre, [batch, -1, 1])
            trajectory_step_fake_pre = tf.reshape(trajectory_step_fake_pre, [batch, -1, 1])

            trajectory_real_predict = tf.concat((trajectory_real_predict, trajectory_step_real_pre), axis=1)
            trajectory_fake_predict = tf.concat((trajectory_fake_predict, trajectory_step_fake_pre), axis=1)

        return trajectory_real_predict, trajectory_fake_predict


def train(hidden_size, n_disc, lambda_balance, learning_rate, l2_regularization, imbalance_kl, z_dims, t_imbalance):
    train_set = np.load('train_x_.npy').reshape(-1, 6, 60)
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)
    test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)

    time_train = train_set[:, :, 0]
    batch_train = time_train.shape[0]
    time_train_ = time_train.reshape([-1, 1])
    scaler = MinMaxScaler()
    scaler.fit(time_train_)
    feature_normalization = scaler.transform(time_train_)
    feature_normalization = feature_normalization.reshape(batch_train, 6, 1)
    train_set = np.concatenate((train_set[:, :, 1:], feature_normalization), axis=2)

    time_test = test_set[:, :, 0]
    batch_test = time_test.shape[0]
    time_test_ = time_test.reshape([-1, 1])
    scaler = MinMaxScaler()
    scaler.fit(time_train_)
    feature_normalization_1 = scaler.transform(time_test_)
    feature_normalization_1 = feature_normalization_1.reshape(batch_test, 6, 1)
    test_set = np.concatenate((test_set[:, :, 1:], feature_normalization_1), axis=2)

    time_step = 6
    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0
    test_set.epoch_completed = 0
    previous_visit = 3
    predicted_visit = 3

    hidden_size = 2**(int(hidden_size))
    z_dims = 2 ** (int(z_dims))
    n_disc = int(n_disc)
    lambda_balance = 10**lambda_balance
    learning_rate = 10**learning_rate
    l2_regularization = 10**l2_regularization
    imbalance_kl = 10 ** imbalance_kl
    t_imbalance = 10 ** t_imbalance

    batch_size = 32
    epochs = 1

    print('previous_visit---{}----predicted—_visit{}'.format(previous_visit, predicted_visit))
    print('----batch_size{}---hidden_size{}---n_disc{}---epochs{}---'
          'lambda_balance{}---learning_rate{}---l2_regularization{}--kl_imbalance{}---z_dims{}---t_imbalance{}'
          .format(batch_size, hidden_size, n_disc, epochs, lambda_balance, learning_rate, l2_regularization,
                  imbalance_kl, z_dims, t_imbalance))
    discriminator = Discriminator(time_step=time_step, batch_size=batch_size, hidden_size=hidden_size,
                                  previous_visit=previous_visit, predicted_visit=predicted_visit)

    encode_context = EncoderHistoryInformation(time_step=time_step, hidden_size=hidden_size,
                                               batch_size=batch_size, previous_visit=previous_visit)

    encode_share = EncodeModify(hidden_size=hidden_size)

    vae_network = VAE(z_dims=z_dims, batch_size=batch_size, encode_dims=hidden_size, predicted_visit=previous_visit,
                      feature_dims=feature_dims, previous_visit=previous_visit)

    prior_network = PriorAndGenerationNetwork(z_dims=z_dims, batch_size=batch_size,
                                              encode_dims=hidden_size, predicted_visit=predicted_visit,
                                              feature_dims=feature_dims, previous_visit=previous_visit)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        while train_set.epoch_completed < 3:
            input_train = train_set.next_batch(batch_size)
            input_x_train = input_train[:, :, 1:]
            input_t_train = input_train[:, :, 0]
            for disc in range(n_disc):
                batch = input_x_train.shape[0]
                # obtain the last visit representation
                # encode_history, encode_history_state = encode_context(input_x_train)
                encode_history_state = tf.zeros(shape=[batch, 2 * hidden_size])
                for t in range(predicted_visit):
                    real_sequence_current_time = input_x_train[:, t, :]
                    encode_history, encode_history_state = encode_share([real_sequence_current_time, encode_history_state])
                # h_i_information = [encode_history, encode_history_state]
                inputs_vae = [encode_history, encode_history_state, input_x_train, input_t_train]
                z_post, z_mean_all_post, z_log_var_all_post, input_reconstruction = vae_network(inputs_vae)

                batch = tf.shape(input_t_train)[0]
                z_all_prior = tf.zeros(shape=[batch, 0, z_dims])
                z_mean_all_prior = tf.zeros(shape=[batch, 0, z_dims])
                z_log_var_all_prior = tf.zeros(shape=[batch, 0, z_dims])
                generated_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
                likelihood_trajectory = tf.zeros(shape=[batch, 0, 1])
                for time in range(predicted_visit):
                    real_sequence_last_time = input_x_train[:, time+previous_visit-1, :]
                    real_sequence_current_time = input_x_train[:, time+previous_visit, :]

                    if time == 0:
                        encode_history_state_recurrence = tf.zeros(shape=[batch, 2*feature_dims])
                        encode_history_recurrence = tf.zeros(shape=[batch, feature_dims])
                    current_time_index = time+previous_visit
                    # real value cannot be derived, change it with shape
                    current_time_index_shape = tf.ones(shape=(current_time_index, 1))

                    inputs_prior = [encode_history, encode_history_state, encode_history_recurrence,
                                    encode_history_state_recurrence, real_sequence_last_time,
                                    real_sequence_current_time,
                                    current_time_index_shape, input_t_train]

                    z_prior, z_mean_prior, z_log_var_prior, generated_next_visit, likelihood_next_visit, encode_history_recurrence, encode_history_state_recurrence = prior_network(inputs_prior)

                    encode_history, encode_history_state = encode_share([real_sequence_current_time, encode_history_state]) # feed the real visit

                    z_mean_prior = tf.reshape(z_mean_prior, [batch, -1, z_dims])
                    z_log_var_prior = tf.reshape(z_log_var_prior, [batch, -1, z_dims])
                    generated_next_visit = tf.reshape(generated_next_visit, [batch, -1, feature_dims])
                    likelihood_next_visit = tf.reshape(likelihood_next_visit, [batch, -1, 1])
                    z_prior = tf.reshape(z_prior, [batch, -1, z_dims])

                    z_mean_all_prior = tf.concat((z_mean_all_prior, z_mean_prior), axis=1)
                    z_log_var_all_prior = tf.concat((z_log_var_all_prior, z_log_var_prior), axis=1)
                    generated_trajectory = tf.concat((generated_trajectory, generated_next_visit), axis=1)
                    likelihood_trajectory = tf.concat((likelihood_trajectory, likelihood_next_visit), axis=1)
                    z_all_prior = tf.concat((z_all_prior, z_prior), axis=1)

                d_real_pre, d_fake_pre = discriminator(input_x_train, generated_trajectory)
                d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
                d_real_pre_ = tf.reshape(d_real_pre, [-1, 1])
                d_real_pre_loss = cross_entropy(tf.ones_like(d_real_pre_), d_real_pre_)
                d_fake_pre_loss = cross_entropy(tf.zeros_like(d_fake_pre_), d_fake_pre_)

                d_loss = d_real_pre_loss + d_fake_pre_loss
                for weight in discriminator.trainable_variables:
                    d_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                gradient_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))

            d_real_pre, d_fake_pre = discriminator(input_x_train, generated_trajectory)
            d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
            mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    input_reconstruction))

            mse_loss_2 = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, :],
                                        generated_trajectory))

            time_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(likelihood_trajectory, 1e-8, 1.0)))
            kl_loss = 0
            for t in range(predicted_visit):
                post_mean = z_mean_all_post[:, t, :]
                post_log_var = z_log_var_all_post[:, t, :]
                prior_mean = z_mean_all_prior[:, t, :]
                prior_log_var = z_log_var_all_prior[:, t, :]

                kld_element = (2 * tf.math.log(prior_log_var) - 2 * tf.math.log(post_log_var) +
                               (tf.math.pow(post_log_var, 2) + tf.math.pow((post_mean - prior_mean), 2)) / tf.math.pow(prior_log_var, 2) - 1)
                kl_loss += tf.reduce_sum(kld_element)

            kl_loss = kl_loss
            # KL = tf.keras.losses.KLDivergence()
            # for m in range(len(z_all_prior)):
            #     posterior_d = z_post[m]
            #     prior_d = z_all_prior[m]
            #     kl_loss = KL(posterior_d, prior_d)
            # kl_loss = tf.reduce_mean(kl_loss)
            gen_loss_1 = cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_)

            print('reconstruction_loss----{}-----mse_loss{}----time_loss{}----gen_loss_1{}------kl_loss{}------'.format(mse_loss, mse_loss_2, time_loss, gen_loss_1, kl_loss))

            # gen_loss = mse_loss + mse_loss_2 + t_imbalance * time_loss + gen_loss_1*lambda_balance + kl_loss
            gen_loss = mse_loss_2

            variables = [var for var in encode_context.trainable_variables]
            for weight in encode_context.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in vae_network.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in  prior_network.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in encode_share.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            gradient_generator = gen_tape.gradient(gen_loss, variables)
            generator_optimizer.apply_gradients(zip(gradient_generator, variables))

            print('开始测试！')
            input_x_test_all = tf.zeros(shape=(0, 6, feature_dims))
            generated_x_test_all = tf.zeros(shape=(0, predicted_visit, feature_dims))
            while test_set.epoch_completed < epochs:
                input_test = test_set.next_batch(batch_size)
                batch_test = input_test.shape[0]
                input_x_test = input_test[:, :, 1:]
                input_t_test = input_test[:, :, 0]
                h_i_test, h_i_information_test = encode_context(input_x_test)
                generated_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                for time in range(predicted_visit):
                    sequence_last_time_test = input_x_test[:, time+previous_visit-1, :]
                    sequence_current_time_test = input_x_test[:, time+previous_visit, :]

                    if time == 0:
                        encode_history_state_test_recurrence = tf.zeros(shape=[batch_test, 2*feature_dims])
                        encode_history_test_recurrence = tf.zeros(shape=(batch_test, feature_dims))
                        h_i_information_test = tf.concat((h_i_information_test[0], h_i_information_test[1]), axis=1)

                    current_time_index_test = time + previous_visit
                    current_time_index_shape = tf.ones(shape=(current_time_index_test, 1))

                    inputs_prior_test = [h_i_test, h_i_information_test,
                                         encode_history_test_recurrence,
                                         encode_history_state_test_recurrence,
                                         sequence_last_time_test,
                                         sequence_current_time_test,
                                         current_time_index_shape, input_t_test]

                    z_prior_test, z_mean_prior_test, z_log_var_prior_test, generated_next_visit_test, likelihood_next_visit_test, encode_history_test_recurrence, encode_history_state_test_recurrence = prior_network(inputs_prior_test)

                    h_i_test, h_i_information_test = encode_share([generated_next_visit_test, h_i_information_test])

                    generated_next_visit_test = tf.reshape(generated_next_visit_test, [batch_test, -1, feature_dims])
                    generated_trajectory_test = tf.concat((generated_trajectory_test, generated_next_visit_test), axis=1)
                input_x_test_all = tf.concat((input_x_test_all, input_x_test), axis=0)
                generated_x_test_all = tf.concat((generated_x_test_all, generated_trajectory_test), axis=0)

            mse_loss_test = tf.reduce_mean(
                    tf.keras.losses.mse(input_x_test_all[:, previous_visit:previous_visit + predicted_visit, :],
                                        generated_x_test_all))

            r_value_all = []
            p_value_all = []
            for r in range(predicted_visit):
                x_ = tf.reshape(input_x_test_all[:, previous_visit + r, :], (-1,))
                y_ = tf.reshape(generated_x_test_all[:, r, :], (-1,))
                r_value_ = stats.pearsonr(x_, y_)
                r_value_all.append(r_value_[0])
                p_value_all.append(r_value_[1])
            print('-----------r_value{}----------'.format(np.mean(r_value_all)))
            print('------------p_value{}-----------'.format(np.mean(p_value_all)))
            print("mse_loss:{}---------".format(mse_loss_test))
            return -1*mse_loss_test


def test_test(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

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
    test_test('7_15_1 修改proposed model.txt')
    GAN_LSTM_BO = BayesianOptimization(
        train, {
            'hidden_size': (3, 5),
            'z_dims': (3, 5),
            'n_disc': (1, 20),
            'lambda_balance': (-6, 0),
            'imbalance_kl': (-6, 0),
            'learning_rate': (-5, -1),
            'l2_regularization': (-5, -1),
            't_imbalance': (-5, -1),
        }
    )
    GAN_LSTM_BO.maximize()
    print(GAN_LSTM_BO.max)


    # mse_all = []
    # for i in range(50):
    #     mse = train(hidden_size=16, n_disc=5, lambda_balance=7.334671684939495e-05,
    #                 learning_rate=0.0024121091663637535, l2_regularization=0.0004291786335604492,
    #                 imbalance_kl=2.842050971659098e-05, z_dims=16, t_imbalance=2.5743381482711544e-05)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))















