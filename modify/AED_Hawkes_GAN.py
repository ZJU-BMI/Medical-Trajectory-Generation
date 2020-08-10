import tensorflow as tf
import numpy as np
import os
from data import DataSet
from utils import Decoder, HawkesProcess, test_test, Encoder
from scipy import stats
from bayes_opt import BayesianOptimization
from ProposedModel import Discriminator


def train(hidden_size, learning_rate, l2_regularization, n_disc, generated_mse_imbalance, generated_loss_imbalance, likelihood_imbalance):
    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

    previous_visit = 1
    predicted_visit = 5

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50

    hidden_size = 2 ** (int(hidden_size))
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    n_disc = int(n_disc)
    generated_mse_imbalance = 10 ** generated_mse_imbalance
    generated_loss_imbalance = 10 ** generated_loss_imbalance
    likelihood_imbalance = 10 ** likelihood_imbalance

    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))

    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}---n_disc---{}'
          'generated_mse_imbalance---{}---generated_loss_imbalance---{}---'
          'likelihood_imbalance---{}'.format(hidden_size, learning_rate, l2_regularization, n_disc,
                                             generated_mse_imbalance, generated_loss_imbalance, likelihood_imbalance))
    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    hawkes_process = HawkesProcess()
    discriminator = Discriminator(previous_visit=previous_visit, predicted_visit=predicted_visit, hidden_size=hidden_size)

    logged = set()
    max_loss = 0.001
    max_pace = 0.0001
    count = 0
    loss = 0
    optimizer_generation = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer_discriminator = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = tf.cast(input_train[:, :, 1:], tf.float32)
        input_t_train = tf.cast(input_train[:, :, 0], tf.float32)
        batch = input_train.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            generated_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            probability_likelihood = tf.zeros(shape=[batch, 0, 1])
            for predicted_visit_ in range(predicted_visit):
                sequence_last_time = input_x_train[:, previous_visit + predicted_visit_ -1, :]
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h

                if predicted_visit_ == 0:
                    decode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                current_time_index_shape = tf.ones(shape=[previous_visit+predicted_visit_])
                intensity_value, likelihood = hawkes_process([input_t_train, current_time_index_shape])
                probability_likelihood = tf.concat((probability_likelihood, tf.reshape(likelihood, [batch, -1, 1])), axis=1)

                generated_next_visit, decode_c, decode_h = decoder_share([sequence_last_time, context_state, decode_c, decode_h*intensity_value])
                generated_trajectory = tf.concat((generated_trajectory, tf.reshape(generated_next_visit, [batch, -1, feature_dims])), axis=1)

            d_real_pre_, d_fake_pre_ = discriminator(input_x_train, generated_trajectory)
            d_real_pre_loss = cross_entropy(tf.ones_like(d_real_pre_), d_real_pre_)
            d_fake_pre_loss = cross_entropy(tf.zeros_like(d_fake_pre_), d_fake_pre_)
            d_loss = d_real_pre_loss + d_fake_pre_loss

            gen_loss = cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_)
            generated_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    generated_trajectory))

            likelihood_loss = tf.reduce_mean(probability_likelihood)

            loss += generated_mse_loss * generated_mse_imbalance + likelihood_loss * likelihood_imbalance + \
                    gen_loss * generated_loss_imbalance

            for weight in discriminator.trainable_variables:
                d_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
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

            mse_generated = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    generated_trajectory))

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
                    decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    sequence_last_time_test = input_x_test[:, previous_visit+predicted_visit_-1, :]

                current_time_index_shape = tf.ones([previous_visit+predicted_visit_])
                intensity_value, likelihood = hawkes_process([input_t_test, current_time_index_shape])
                generated_next_visit,  decode_c_test, decode_h_test = decoder_share([sequence_last_time_test, context_state_test, decode_c_test, decode_h_test*intensity_value])
                generated_trajectory_test = tf.concat((generated_trajectory_test, tf.reshape(generated_next_visit, [batch_test, -1, feature_dims])), axis=1)
                sequence_last_time_test = generated_next_visit

            mse_generated_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))
            mae_generated_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))

            r_value_all = []
            p_value_all = []

            for r in range(predicted_visit):
                x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
                if (y_.numpy() == np.zeros_like(y_)).all():
                    r_value_ = [0.0, 0.0]
                else:
                    r_value_ = stats.pearsonr(x_, y_)
                r_value_all.append(r_value_[0])
                p_value_all.append(r_value_[1])

            print('epoch ---{}---train_mse_generated---{}---'
                  'test_mse---{}---test_mae---{}---'
                  'r_value_test---{}---count---{}'.format(train_set.epoch_completed, generated_mse_loss,
                                                          mse_generated_test, mae_generated_test,
                                                          np.mean(r_value_all), count))

    tf.compat.v1.reset_default_graph()
    # return mse_generated_test, mae_generated_test, np.mean(r_value_all)
    return -1 * mse_generated_test


if __name__ == '__main__':
    test_test('AED_Hawkes_GAN_HF_test__1_5_重新训练_7_25.txt')
    BO = BayesianOptimization(
        train, {
            'hidden_size': (5, 8),
            'n_disc': (1, 10),
            'learning_rate': (-5, 1),
            'l2_regularization': (-5, 1),
            'generated_mse_imbalance': (-6, 1),
            'likelihood_imbalance': (-6, 1),
            'generated_loss_imbalance': (-6, 1),

        }
    )
    BO.maximize()
    print(BO.max)
    # mse_all = []
    # r_value_all = []
    # mae_all = []
    # for i in range(50):
    #     mse, mae, r_value = train(hidden_size=128,
    #                               learning_rate=0.004481554158981572,
    #                               l2_regularization=0.0063829369825288244,
    #                               n_disc=3,
    #                               generated_mse_imbalance=1.9699616185956445e-05,
    #                               generated_loss_imbalance=0.061039544920758,
    #                               likelihood_imbalance=0.018446327803507835)
    #     mse_all.append(mse)
    #     r_value_all.append(r_value)
    #     mae_all.append(mae)
    #     print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
    #           "r_value_std {}----mse_all_std  {}  mae_std {}".
    #           format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
    #                  np.std(r_value_all), np.std(mse_all), np.std(mae_all)))



















