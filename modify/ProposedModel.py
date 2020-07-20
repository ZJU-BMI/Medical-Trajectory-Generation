import tensorflow as tf
import numpy as np
from data import DataSet
import os
import warnings
from tensorflow_core.python.keras.models import Model
from test import Post, Prior, HawkesProcess, Encoder, Decoder, test_test
from scipy import stats
from bayes_opt import BayesianOptimization


warnings.filterwarnings(action='once')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Discriminator(Model):
    def __init__(self, hidden_size, previous_visit, predicted_visit):
        super().__init__(name='discriminator')
        self.hidden_size = hidden_size
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit

        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
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
            next_real = real_input[:, index + self.previous_visit, :]
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

            trajectory_step_real_pre = self.dense3(trajectory_step_real_pre)
            trajectory_step_fake_pre = self.dense3(trajectory_step_fake_pre)

            trajectory_step_real_pre = self.dense4(trajectory_step_real_pre)
            trajectory_step_fake_pre = self.dense4(trajectory_step_fake_pre)

            trajectory_step_real_pre = tf.reshape(trajectory_step_real_pre, [batch, -1, 1])
            trajectory_step_fake_pre = tf.reshape(trajectory_step_fake_pre, [batch, -1, 1])

            trajectory_real_predict = tf.concat((trajectory_real_predict, trajectory_step_real_pre), axis=1)
            trajectory_fake_predict = tf.concat((trajectory_fake_predict, trajectory_step_fake_pre), axis=1)

        return trajectory_real_predict, trajectory_fake_predict


def train(hidden_size, z_dims, l2_regularization, learning_rate, n_disc, generated_mse_imbalance, generated_loss_imbalance, kl_imbalance, reconstruction_mse_imbalance, likelihood_imbalance):
    train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    # train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/HF_test_.npy').reshape(-1, 6, 30)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50

    hidden_size = 2**(int(hidden_size))
    z_dims = 2**(int(z_dims))
    l2_regularization = 10 ** l2_regularization
    learning_rate = 10 ** learning_rate
    n_disc = int(n_disc)
    generated_mse_imbalance = 10 ** generated_mse_imbalance
    generated_loss_imbalance = 10 ** generated_loss_imbalance
    kl_imbalance = 10 ** kl_imbalance
    reconstruction_mse_imbalance = 10 ** reconstruction_mse_imbalance
    likelihood_imbalance = 10 ** likelihood_imbalance

    print('hidden_size---{}---z_dims---{}---l2_regularization---{}---learning_rate---{}--n_disc---{}-'
          'generated_mse_imbalance---{}---generated_loss_imbalance---{}---'
          'kl_imbalance---{}---reconstruction_mse_imbalance---{}---'
          'likelihood_imbalance---{}'.format(hidden_size, z_dims, l2_regularization,
                                             learning_rate, n_disc, generated_mse_imbalance,
                                             generated_loss_imbalance, kl_imbalance,
                                             reconstruction_mse_imbalance, likelihood_imbalance))

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    discriminator = Discriminator(predicted_visit=predicted_visit, hidden_size=hidden_size, previous_visit=previous_visit)

    post_net = Post(z_dims=z_dims)
    prior_net = Prior(z_dims=z_dims)

    hawkes_process = HawkesProcess()

    loss = 0
    count = 0
    optimizer_generation = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer_discriminator = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    logged = set()
    max_loss = 0.001
    max_pace = 0.0001

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = tf.cast(input_train[:, :, 1:], tf.float32)
        input_t_train = tf.cast(input_train[:, :, 0], tf.float32)
        batch = input_train.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            generated_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            probability_likelihood = tf.zeros(shape=[batch, 0, 1])
            reconstructed_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            z_mean_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_log_var_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_mean_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            z_log_var_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            for predicted_visit_ in range(predicted_visit):
                sequence_last_time = input_x_train[:, previous_visit + predicted_visit_ - 1, :]
                sequence_current_time = input_x_train[:, previous_visit+predicted_visit_, :]
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h  # h_i
                encode_c, encode_h = encode_share([sequence_current_time, encode_c, encode_h]) # h_(i+1)

                if predicted_visit_ == 0:
                    decode_c_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    decode_c_reconstruction = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_reconstruction = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                z_post, z_mean_post, z_log_var_post = post_net([context_state, encode_h])
                z_prior, z_mean_prior, z_log_var_prior = prior_net(context_state)

                current_time_index_shape = tf.ones(shape=[previous_visit+predicted_visit_])
                condition_value, likelihood = hawkes_process([input_t_train, current_time_index_shape])
                probability_likelihood = tf.concat((probability_likelihood, tf.reshape(likelihood, [batch, -1, 1])), axis=1)
                # generation
                generated_next_visit, decode_c_generate, decode_h_generate = decoder_share([z_prior, context_state, sequence_last_time, decode_c_generate, decode_h_generate*condition_value])
                # reconstruction
                reconstructed_next_visit, decode_c_reconstruction, decode_h_reconstruction = decoder_share([z_post, context_state, sequence_last_time, decode_c_reconstruction, decode_h_reconstruction*condition_value])

                reconstructed_trajectory = tf.concat((reconstructed_trajectory, tf.reshape(reconstructed_next_visit, [batch, -1, feature_dims])), axis=1)
                generated_trajectory = tf.concat((generated_trajectory, tf.reshape(generated_next_visit, [batch, -1, feature_dims])), axis=1)

                z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)

                z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)
                z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)

            d_real_pre_, d_fake_pre_ = discriminator(input_x_train, generated_trajectory)
            d_real_pre_loss = cross_entropy(tf.ones_like(d_real_pre_), d_real_pre_)
            d_fake_pre_loss = cross_entropy(tf.zeros_like(d_fake_pre_), d_fake_pre_)
            d_loss = d_real_pre_loss + d_fake_pre_loss

            gen_loss = cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_)
            generated_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    generated_trajectory))
            reconstructed_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    reconstructed_trajectory))

            std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
            std_prior = tf.math.sqrt(tf.exp(z_log_var_prior_all))

            kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(
                tf.maximum(std_post, 1e-9))
                                     + (tf.square(std_post) + (tf.square(z_mean_post_all - z_mean_prior_all)) /
                                        (tf.maximum(tf.square(std_prior), 1e-9))) - 1)
            kl_loss = tf.reduce_mean(kl_loss_element)

            likelihood_loss = tf.reduce_mean(probability_likelihood)

            loss += generated_mse_loss * generated_mse_imbalance +\
                    reconstructed_mse_loss * reconstruction_mse_imbalance + \
                    kl_loss * kl_imbalance + likelihood_loss * likelihood_imbalance \
                    + gen_loss * generated_loss_imbalance

            for weight in discriminator.trainable_variables:
                d_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in post_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in prior_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in hawkes_process.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

        for disc in range(n_disc):
            gradient_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            optimizer_discriminator.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))

        gradient_gen = gen_tape.gradient(loss, variables)
        optimizer_generation.apply_gradients(zip(gradient_gen, variables))

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)
            loss_pre = generated_mse_loss

            mse_generated = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory))

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

            batch_test = test_set.shape[0]
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
                        encode_c_test, encode_h_test = encode_share([generated_trajectory_test[:, i, :], encode_c_test, encode_h_test])

                context_state_test = encode_h_test

                if predicted_visit_ == 0:
                    decode_c_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    decode_h_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    sequence_last_time_test = input_x_test[:, previous_visit+predicted_visit_-1, :]

                z_prior_test, z_mean_prior_test, z_log_var_prior_test = prior_net(context_state_test)
                current_time_index_shape_test = tf.ones([previous_visit+predicted_visit_])
                intensity_value_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])

                generated_next_visit_test, decode_c_generate_test, decode_h_generate_test = decoder_share([z_prior_test, context_state_test, sequence_last_time_test, decode_c_generate_test, decode_h_generate_test*intensity_value_test])
                generated_trajectory_test = tf.concat((generated_trajectory_test, tf.reshape(generated_next_visit_test, [batch_test, -1, feature_dims])), axis=1)
                sequence_last_time_test = generated_next_visit_test

            mse_generated_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))
            mae_generated_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))

            r_value_all = []
            p_value_all = []

            for r in range(predicted_visit):
                x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
                if y_.numpy().all() == np.zeros_like(y_).all():
                    r_value_ = [0.0, 0.0]
                else:
                    r_value_ = stats.pearsonr(x_, y_)
                r_value_all.append(r_value_[0])
                p_value_all.append(r_value_[1])

            print('epoch ---{}---train_mse_generated---{}---'
                  'train_mse_reconstruct---{}---train_kl---{}---'
                  'test_mse---{}---test_mae---{}---'
                  'r_value_test---{}---count---{}'.format(train_set.epoch_completed, generated_mse_loss,
                                                          reconstructed_mse_loss, kl_loss,
                                                          mse_generated_test, mae_generated_test,
                                                          np.mean(r_value_all), count))

    tf.compat.v1.reset_default_graph()
    # return mse_generated_test, mae_generated_test, np.mean(r_value_all)
    return -1 * mse_generated_test


if __name__ == '__main__':
    test_test('VAE_Hawkes_GAN_青光眼_train_3_3_7_20.txt')
    BO = BayesianOptimization(
        train, {
            'hidden_size': (5, 8),
            'z_dims': (5, 8),
            'n_disc': (1, 10),
            'learning_rate': (-5, 1),
            'l2_regularization': (-5, 1),
            'kl_imbalance':  (-6, 1),
            'reconstruction_mse_imbalance': (-6, 1),
            'generated_mse_imbalance': (-6, 1),
            'likelihood_imbalance': (-6, 1),
            'generated_loss_imbalance': (-6, 1),

        }
    )
    BO.maximize()
    print(BO.max)































