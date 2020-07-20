import tensorflow as tf
import numpy as np
import sys
import os
from data import DataSet
from utils import Decoder, HawkesProcess, test_test, Encoder
from scipy import stats
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings(action='once')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train(hidden_size, l2_regularization, learning_rate, generated_imbalance, likelihood_imbalance):
    train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50

    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # generated_imbalance = 10 ** generated_imbalance
    # likelihood_imbalance = 10 ** likelihood_imbalance

    print('hidden_size----{}---'
          'l2_regularization---{}---'
          'learning_rate---{}---'
          'generated_imbalance---{}---'
          'likelihood_imbalance---{}'.
          format(hidden_size, l2_regularization, learning_rate,
                 generated_imbalance, likelihood_imbalance))

    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    encode_share = Encoder(hidden_size=hidden_size)
    hawkes_process = HawkesProcess()

    logged = set()
    max_loss = 0.01
    max_pace = 0.001
    loss = 0

    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        batch = input_train.shape[0]
        input_x_train = tf.cast(input_train[:, :, 1:], tf.float32)
        input_t_train = tf.cast(input_train[:, :, 0], tf.float32)

        with tf.GradientTape() as tape:
            predicted_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            likelihood_all = tf.zeros(shape=[batch, 0, 1])
            for predicted_visit_ in range(predicted_visit):
                sequence_time_last_time = input_x_train[:, previous_visit+predicted_visit_-1, :]
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
                current_time_index_shape = tf.ones(shape=[predicted_visit_+previous_visit])
                condition_intensity, likelihood = hawkes_process([input_t_train, current_time_index_shape])
                likelihood_all = tf.concat((likelihood_all, tf.reshape(likelihood, [batch, -1, 1])), axis=1)
                generated_next_visit, decode_c, decode_h = decoder_share([sequence_time_last_time, context_state, decode_c, decode_h*condition_intensity])
                predicted_trajectory = tf.concat((predicted_trajectory, tf.reshape(generated_next_visit, [batch, -1, feature_dims])), axis=1)

            mse_generated_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory))
            mae_generated_loss = tf.reduce_mean(tf.keras.losses.mae(input_x_train[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory))
            likelihood_loss = tf.reduce_mean(likelihood_all)

            loss += mse_generated_loss * generated_imbalance + likelihood_loss * likelihood_imbalance

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in hawkes_process.trainable_variables:
                variables.append(weight)
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)

                loss_pre = mse_generated_loss
                mse_generated_loss = tf.reduce_mean(
                    tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                        predicted_trajectory))

                loss_diff = loss_pre - mse_generated_loss

                if max_loss < mse_generated_loss:
                    count = 0
                else:
                    if max_pace < loss_diff:
                        count = 0

                    else:
                        count += 1
                if count > 9:
                    break

                input_x_test = tf.cast(test_set[:, :, 1:], tf.float32)
                input_t_test = tf.cast(test_set[:, :, 0], tf.float32)

                batch_test = input_x_test.shape[0]
                predicted_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            encode_c, encode_h_test = encode_share([predicted_trajectory_test[:, i, :], encode_c_test, encode_h_test])
                    context_state_test = encode_h_test

                    if predicted_visit_ == 0:
                        decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        sequence_time_last_time_test = input_x_test[:, predicted_visit_+previous_visit-1, :]

                    current_time_index_shape_test = tf.ones(shape=[previous_visit+predicted_visit_])
                    condition_intensity_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])

                    sequence_next_visit_test, decode_c_test, decode_h_test = decoder_share([sequence_time_last_time_test, context_state_test, decode_c_test, decode_h_test*condition_intensity_test])
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(sequence_next_visit_test, [batch_test, -1, feature_dims])), axis=1)
                    sequence_time_last_time_test = sequence_next_visit_test

                mse_generated_loss_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
                mae_generated_loss_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))

                r_value_all = []
                p_value_all = []

                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(predicted_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                print("epoch  {}---train_mse_generate {}- - "
                      "mae_generated_loss--{}--test_mse {}--test_mae  "
                      "{}----r_value {}---count {}".format(train_set.epoch_completed,
                                                           mse_generated_loss,
                                                           mae_generated_loss,
                                                           mse_generated_loss_test,
                                                           mae_generated_loss_test,
                                                           np.mean(r_value_all),
                                                           count))
    tf.compat.v1.reset_default_graph()
    return mse_generated_loss_test, mae_generated_loss_test, np.mean(r_value_all)
    # return -1 * mse_generated_loss_test


if __name__ == '__main__':
    test_test('AED_Hawkes_青光眼_test_3_3_7_18.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, 1),
    #         'l2_regularization': (-5, 1),
    #         'generated_imbalance': (-6, 1),
    #         'likelihood_imbalance': (-6, 1)
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    mse_all = []
    r_value_all = []
    mae_all = []
    for i in range(50):
        mse, mae, r_value = train(hidden_size=128,
                                  learning_rate=0.008823066301767246,
                                  l2_regularization=3.017286348529315e-05,
                                  generated_imbalance=0.47383867868002144,
                                  likelihood_imbalance=0.0030426563595897745)
        mse_all.append(mse)
        r_value_all.append(r_value)
        mae_all.append(mae)
        print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
              "r_value_std {}----mse_all_std  {}  mae_std {}".
              format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
                     np.std(r_value_all), np.std(mse_all),np.std(mae_all)))








