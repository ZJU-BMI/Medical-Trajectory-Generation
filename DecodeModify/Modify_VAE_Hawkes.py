import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet
from bayes_opt import BayesianOptimization
import scipy.stats as stats
from EncoderDecoder import test_test
import os
import numpy as np

import warnings
warnings.filterwarnings(action='once')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 所有共享的encoder 输入x_i state, 输出新的state
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


# 输入z_i_1, 输出解码x_i_1
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
        decode_input, decode_c, decode_h = input_x
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decode(decode_input, state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]


# 输入h_i, 输出z_i
class Prior(Model):
    def __init__(self, z_dims):
        super().__init__(name='prior_net')
        self.z_dims = z_dims
        self.dense1 = tf.keras.layers.Dense(z_dims)
        self.dense2 = tf.keras.layers.Dense(z_dims)
        self.dense3 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)  # obtain z hidden
        # obtain z_mean, z_log_var
        self.dense4 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)
        self.dense5 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)

    def call(self, input_x):
        hidden_1 = self.dense1(input_x)
        hidden_2 = self.dense2(hidden_1)
        hidden_3 = self.dense3(hidden_2)

        z_mean = self.dense4(hidden_3)
        z_log_var = self.dense5(hidden_3)

        z = self.reparameterize(z_mean, z_log_var, self.z_dims)

        return z, z_mean, z_log_var

    def reparameterize(self, mu, log_var, z_dims):
        batch = tf.shape(mu)[0]
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        return z


# 输入h_(i+1) and h_i 输出z_(i+1)
class Post(Model):
    def __init__(self, z_dims):
        super().__init__(name='post_net')
        self.z_dims = z_dims
        self.dense1 = tf.keras.layers.Dense(z_dims)
        self.dense2 = tf.keras.layers.Dense(z_dims)
        self.dense3 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)  # obtain z hidden
        # obtain z_mean, z_log_var
        self.dense4 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)
        self.dense5 = tf.keras.layers.Dense(z_dims, activation=tf.nn.sigmoid)

    def call(self, input_x):
        h_i, h_i_1 = input_x
        hidden = tf.concat((h_i, h_i_1), axis=1)

        hidden_1 = self.dense1(hidden)
        hidden_2 = self.dense2(hidden_1)
        hidden_3 = self.dense3(hidden_2)

        z_mean = self.dense4(hidden_3)
        z_log_var = self.dense5(hidden_3)

        z = self.reparameterize(z_mean, z_log_var, self.z_dims)

        return z, z_mean, z_log_var

    def reparameterize(self, mu, log_var, z_dims):
        batch = tf.shape(mu)[0]
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        return z


# 输入input_t, 输出lambda_(t_(i+1)), f(t_(i+1))
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


def kl_loss(z_mean_post, log_var_post, z_mean_prior, log_var_prior):
    std_post = tf.math.sqrt(tf.exp(log_var_post))
    std_prior = tf.math.sqrt(tf.exp(log_var_prior))
    kl_loss_element = (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(tf.maximum(std_post, 1e-9)) +
                       (tf.math.pow(std_post, 2) +
                       tf.math.pow((z_mean_post - z_mean_prior), 2)) / tf.maximum(tf.math.pow(z_mean_prior, 2), 1e-9)-1)
    return 0.5 * kl_loss_element


def train(hidden_size, z_dims, l2_regularization, learning_rate, kl_imbalance, reconstruction_imbalance, generated_mse_imbalance, likelihood_imbalance):
    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50
    #
    # hidden_size = 2 ** (int(hidden_size))
    # z_dims = 2 ** (int(z_dims))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # kl_imbalance = 10 ** kl_imbalance
    # reconstruction_imbalance = 10 ** reconstruction_imbalance
    # generated_mse_imbalance = 10 ** generated_mse_imbalance
    # likelihood_imbalance = 10 ** likelihood_imbalance

    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))
    print('hidden_size{}----z_dims{}------learning_rate{}----l2_regularization{}---'
          'kl_imbalance{}----reconstruction_imbalance '
          ' {}----generated_mse_imbalance{}----'.format(hidden_size, z_dims,
                                                        learning_rate,
                                                        l2_regularization,
                                                        kl_imbalance,
                                                        reconstruction_imbalance,
                                                        generated_mse_imbalance))

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    prior_net = Prior(z_dims=z_dims)
    post_net = Post(z_dims=z_dims)
    hawkes_process = HawkesProcess()

    logged = set()
    max_loss = 0.001
    max_pace = 0.0001

    loss = 0
    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = tf.cast(input_train[:, :, 1:], tf.float32)
        input_t_train = tf.cast(input_train[:, :, 0], tf.float32)
        batch = input_x_train.shape[0]

        with tf.GradientTape() as tape:
            generated_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            reconstruction_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            z_log_var_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_mean_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_mean_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            z_log_var_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            probability_likelihood = tf.zeros(shape=[batch, 0, 1])

            for predicted_visit_ in range(predicted_visit):
                sequence_time_current_time = input_x_train[:, predicted_visit_+previous_visit, :]
                sequence_time_last_time = input_x_train[:, predicted_visit_+previous_visit-1, :]
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h

                if predicted_visit_ == 0:
                    decode_c_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    decode_c_reconstruct = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_reconstruct = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                current_time_index_shape = tf.ones(shape=[predicted_visit_ + previous_visit])
                condition_intensity, likelihood = hawkes_process([input_t_train, current_time_index_shape])

                z_prior, z_mean_prior, z_log_var_prior = prior_net(context_state*condition_intensity)
                encode_c, encode_h = encode_share([sequence_time_current_time, encode_c, encode_h])
                z_post, z_mean_post, z_log_var_post = post_net([context_state*condition_intensity, encode_h])

                generated_next_visit, decode_c_generate, decode_h_generate = decoder_share([z_prior, decode_c_generate, decode_h_generate])
                likelihood = tf.reshape(likelihood, [batch, -1, 1])
                probability_likelihood = tf.concat((probability_likelihood, likelihood), axis=1)
                generated_next_visit = tf.reshape(generated_next_visit, [batch, -1, feature_dims])
                generated_trajectory = tf.concat((generated_trajectory, generated_next_visit), axis=1)

                construct_next_visit, decode_c_reconstruct, decode_h_reconstruct = decoder_share([z_post, decode_c_reconstruct, decode_h_reconstruct])
                construct_next_visit = tf.reshape(construct_next_visit, [batch, -1, feature_dims])
                reconstruction_trajectory = tf.concat((reconstruction_trajectory, construct_next_visit), axis=1)

                z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)
                z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)

                z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)

            mse_reconstruction = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :], reconstruction_trajectory))
            mse_generated = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :], reconstruction_trajectory))

            std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
            std_prior = tf.math.sqrt(tf.exp(z_log_var_prior_all))

            kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(tf.maximum(std_post,
                                                                                                               1e-9)) +
                                     (tf.math.pow(std_post, 2) + tf.math.pow((z_mean_post_all - z_mean_prior_all), 2)) /
                                     tf.maximum(tf.math.pow(std_prior, 2), 1e-9) - 1)
            kl_loss_all = tf.reduce_mean(kl_loss_element)
            # print('kl_loss---{}'.format(kl_loss_all))

            likelihood_loss = tf.reduce_mean(probability_likelihood)

            loss += mse_reconstruction * reconstruction_imbalance + mse_generated * generated_mse_imbalance + kl_loss_all * kl_imbalance + likelihood_loss * likelihood_imbalance

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in prior_net.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in post_net.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in hawkes_process.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            tape.watch(variables)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                loss_pre = mse_generated

                mse_generated = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :], reconstruction_trajectory))
                loss_diff = loss_pre - mse_generated

                if mse_generated > max_loss:
                    count = 0

                else:
                    if loss_diff > max_pace:
                        count = 0

                    else:
                        count += 1
                if count > 9:
                    break

                input_x_test = tf.cast(test_set[:, :, 1:], tf.float32)
                input_t_test = tf.cast(test_set[:, :, 0], tf.float32)
                batch_test = input_x_test.shape[0]
                generated_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            sequence_input_t = generated_trajectory_test[:, i, :]
                            encode_c_test, encode_h_test = encode_share([sequence_input_t, encode_c_test, encode_h_test])

                    context_state_test = encode_h_test

                    if predicted_visit_ == 0:
                        decode_c_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        sequence_last_time_test = input_x_test[:, predicted_visit_+previous_visit, :]

                    current_time_index_shape_test = tf.ones(shape=[previous_visit+predicted_visit_])
                    condition_intensity_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])
                    z_prior_test, z_mean_prior_test, z_log_var_prior = prior_net(context_state_test*condition_intensity_test)

                    sequence_next_visit_test, decode_c_generate_test, decode_h_generate_test = decoder_share([z_prior_test, decode_c_generate_test, decode_h_generate_test])
                    sequence_last_time_test = sequence_next_visit_test
                    sequence_next_visit_test = tf.reshape(sequence_last_time_test, [batch_test, -1, feature_dims])

                    generated_trajectory_test = tf.concat((generated_trajectory_test, sequence_next_visit_test), axis=1)

                mse_generated_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))
                mae_generated_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))

                r_value_all = []
                p_value_all = []

                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                print("epoch  {}---train_mse_generate {}--train_reconstruct {}--train_kl "
                      "{}--test_mse {}--test_mae  {}----r_value {}---count {}".format(train_set.epoch_completed,
                                                                                      mse_generated,
                                                                                      mse_reconstruction,
                                                                                      kl_loss_all,
                                                                                      mse_generated_test,
                                                                                      mae_generated_test,
                                                                                      np.mean(r_value_all),
                                                                                      count))

        # if mse_generated_test < 0.008 and train_set.epoch_completed > 40:
        #     checkpoint_encode_share = tf.train.Checkpoint(encode_share=encode_share)
        #     path = './model/encoder_share/' + str(train_set.epoch_completed) + '.ckpt'
        #     path_encoder = checkpoint_encode_share.save(path)
        #     print(path_encoder)
        #
        #     checkpoint_decode_share = tf.train.Checkpoint(decoder_share=decoder_share)
        #     path = './model/decoder_share/' + str(train_set.epoch_completed) + '.ckpt'
        #     path_decoder = checkpoint_decode_share.save(path)
        #     print(path_decoder)
        #
        #     checkpoint_post = tf.train.Checkpoint(post_net=post_net)
        #     path = './model/checkpoint_post/' + str(train_set.epoch_completed) + '.ckpt'
        #     path_post = checkpoint_post.save(path)
        #     print(path_post)
        #
        #     checkpoint_prior = tf.train.Checkpoint(prior_net=prior_net)
        #     path = './model/checkpoint_prior/' + str(train_set.epoch_completed) + '.ckpt'
        #     path_prior = checkpoint_prior.save(path)
        #     print(path_prior)
        #
        #     checkpoint_hawkes = tf.train.Checkpoint(hawkes_process=hawkes_process)
        #     path = './model/checkpoint_hawkes/' + str(train_set.epoch_completed) + '.ckpt'
        #     path_hawkes = checkpoint_hawkes.save(path)
        #     print(path_hawkes)

    tf.compat.v1.reset_default_graph()
    return mse_generated_test, mae_generated_test, np.mean(r_value_all)
    # return -1 * mse_generated_test


def test():

    train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0

    hidden_size = 64
    z_dims = 64

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    prior_net = Prior(z_dims=z_dims)
    post_net = Post(z_dims=z_dims)
    hawkes_process = HawkesProcess()

    checkpoint_encode_share = tf.train.Checkpoint(encode_share=encode_share)
    checkpoint_encode_share.restore(tf.train.latest_checkpoint('./model/encoder_share/46.ckpt'))

    checkpoint_decode_share = tf.train.Checkpoint(decoder_share=decoder_share)
    checkpoint_decode_share.restore(tf.train.latest_checkpoint('./model/decoder_share/46.ckpt'))

    checkpoint_post = tf.train.Checkpoint(post_net=post_net)
    checkpoint_post.restore(tf.train.latest_checkpoint('./model/checkpoint_post/46.ckpt'))

    checkpoint_prior = tf.train.Checkpoint(prior_net=prior_net)
    checkpoint_prior.restore(tf.train.latest_checkpoint('./model/checkpoint_prior/46.ckpt'))

    checkpoint_hawkes = tf.train.Checkpoint(hawkes_process=hawkes_process)
    checkpoint_hawkes.restore(tf.train.latest_checkpoint('./model/checkpoint_hawkes/46.ckpt'))

    input_x_test = tf.cast(test_set[:, :, 1:], tf.float32)
    input_t_test = tf.cast(test_set[:, :, 0], tf.float32)
    batch_test = input_x_test.shape[0]
    generated_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
    for predicted_visit_ in range(predicted_visit):
        for previous_visit_ in range(previous_visit + predicted_visit_):
            sequence_time_test = input_x_test[:, previous_visit_, :]
            if previous_visit_ == 0:
                encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
            encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

        if predicted_visit_ != 0:
            for i in range(predicted_visit_):
                sequence_input_t = generated_trajectory_test[:, i, :]
                encode_c_test, encode_h_test = encode_share([sequence_input_t, encode_c_test, encode_h_test])

        context_state_test = encode_h_test
        z_prior_test, z_mean_prior_test, z_log_var_prior = prior_net(context_state_test)

        if predicted_visit_ == 0:
            decode_c_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
            decode_h_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
            sequence_last_time_test = input_x_test[:, predicted_visit_ + previous_visit, :]

        current_time_index_shape_test = tf.ones(shape=[previous_visit + predicted_visit_])

        condition_intensity_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])

        sequence_next_visit_test, decode_c_generate_test, decode_h_generate_test = decoder_share([z_prior_test, context_state_test, sequence_last_time_test, decode_c_generate_test, decode_h_generate_test * condition_intensity_test])
        sequence_last_time_test = sequence_next_visit_test
        sequence_next_visit_test = tf.reshape(sequence_last_time_test, [batch_test, -1, feature_dims])

        generated_trajectory_test = tf.concat((generated_trajectory_test, sequence_next_visit_test), axis=1)

    mse_generated_test = tf.reduce_mean(
        tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit + predicted_visit, :],
                            generated_trajectory_test))
    mae_generated_test = tf.reduce_mean(
        tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :],
                            generated_trajectory_test))

    r_value_all = []
    p_value_all = []

    for r in range(predicted_visit):
        x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
        y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
        r_value_ = stats.pearsonr(x_, y_)
        r_value_all.append(r_value_[0])
        p_value_all.append(r_value_[1])

    return mse_generated_test, mae_generated_test, np.mean(r_value_all)


if __name__ == '__main__':
    test_test('VAE_Hawkes_sigmoid_HF_test__3_3_重新训练_8_7_修改解码器输入.txt')

    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'z_dims': (5, 8),
    #         'learning_rate': (-5, 1),
    #         'l2_regularization': (-5, 1),
    #         'kl_imbalance':  (-6, 1),
    #         'reconstruction_imbalance': (-6, 1),
    #         'generated_mse_imbalance': (-6, 1),
    #         'likelihood_imbalance': (-6, 1)
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    mse_all = []
    r_value_all = []
    mae_all = []
    for i in range(50):
        mse, mae, r_value = train(hidden_size=64,
                                  learning_rate=0.0064155641701513205,
                                  l2_regularization=1.274353936947241e-5,
                                  z_dims=128,
                                  kl_imbalance=0.01000828377541689,
                                  generated_mse_imbalance=9.12130343290256e-5 ,
                                  reconstruction_imbalance=0.004736464537618645,
                                  likelihood_imbalance=10 **(-2.4506935847825364))
        mse_all.append(mse)
        r_value_all.append(r_value)
        mae_all.append(mae)
        print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
              "r_value_std {}----mse_all_std  {}  mae_std {}".
              format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
                     np.std(r_value_all), np.std(mse_all),np.std(mae_all)))

    # mse_all = []
    # mae_all = []
    # r_value_all = []
    # for i in range(50):
    #     mse, mae, r_value = test()
    #     mse_all.append(mse)
    #     mae_all.append(mae)
    #     r_value_all.append(r_value)
    #     print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
    #           "r_value_std {}----mse_all_std  {}  mae_std {}".
    #           format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
    #                  np.std(r_value_all), np.std(mse_all), np.std(mae_all)))















