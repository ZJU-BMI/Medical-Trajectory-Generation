import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet
from bayes_opt import BayesianOptimization
import scipy.stats as stats
import os
import numpy as np
from EncoderDecoder import test_test

import warnings
warnings.filterwarnings(action='once')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Encoder(Model):
    def __init__(self, hidden_size):
        super().__init__(name='Encoder_net')
        self.hidden_size = hidden_size
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_data):
        feature, time, encode_c, encode_h = input_data
        state = [encode_c, encode_h]
        inputs = tf.concat((feature, time), axis=1)
        output, state = self.LSTM_Cell(inputs, state)
        return state[0], state[1]


class Decode(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_net')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.sigmoid)

    def call(self, input_data):
        hidden_state, decode_c, decode_h = input_data
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell(hidden_state, state)
        output_ = self.dense1(output)
        output_ = self.dense2(output_)
        output_ = self.dense3(output_)
        return output_, state[0], state[1]


class PointProcess(Model):
    def __init__(self):
        super().__init__(name='point_process')
        self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)  # dense层已经包含了bias

    def build(self, input_shape):
        shape_weight = tf.TensorShape((1, 1))
        self.weight_w = self.add_weight(name='w_t',
                                        shape=shape_weight,
                                        initializer='uniform',
                                        trainable=True)

        super(PointProcess, self).build(input_shape)

    def call(self, input_data):
        hidden_state, time_interval = input_data
        v_h_b = self.dense1(hidden_state)
        log_f_t_i = v_h_b + self.weight_w * time_interval + \
                    (tf.math.exp(v_h_b) - tf.math.exp(v_h_b + self.weight_w * time_interval)) /self.weight_w
        return log_f_t_i


def calculate_point_loss(v_t, h_t, w_t, interval, b_t):
    m = tf.matmul(h_t, v_t) + b_t
    log_f_t_1 = m + w_t * interval + (tf.math.exp(m)- tf.math.exp(m+ w_t* interval)) / w_t
    return log_f_t_1


def train(hidden_size, learning_rate, l2_regularization, point_process_imbalance):

    train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2] -1

    train_set = DataSet(train_set)

    batch_size = 64
    epochs = 50
    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # point_process_imbalance = 10 ** point_process_imbalance

    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))

    print('hidden_size{}-----learning_rate{}----l2_regularization{}---'
          'imbalance--{}-'.format(hidden_size, learning_rate,
                                  l2_regularization,
                                  point_process_imbalance))

    train_set.epoch_completed = 0

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decode(hidden_size=hidden_size, feature_dims=feature_dims)
    point_process = PointProcess()

    logged = set()

    max_loss = 0.01
    max_pace = 0.0001

    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size)
        batch = input_train.shape[0]
        input_x_train = input_train[:, :, 1:]
        input_t_train = tf.reshape(input_train[:, :, 0], [batch, -1, 1])
        with tf.GradientTape() as tape:
            predicted_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            log_f_t_i_all = tf.zeros(shape=[batch, 0])
            for predicted_visit_ in range(predicted_visit):
                time_interval = input_t_train[:, previous_visit+predicted_visit_, :] - input_t_train[:, previous_visit+predicted_visit_-1, :]
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    feature = input_x_train[:, previous_visit_, :]
                    time = input_t_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    encode_c, encode_h = encode_share([feature, time, encode_c, encode_h])
                output = encode_h
                if predicted_visit_ == 0:
                    decode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                log_f_t_i = point_process([output, time_interval])
                log_f_t_i_all = tf.concat((log_f_t_i_all, tf.reshape(log_f_t_i, [batch, -1])), axis=1)
                generated_next_visit, decode_c, decode_h = decoder_share([output, decode_c, decode_h])
                predicted_trajectory = tf.concat((predicted_trajectory, tf.reshape(generated_next_visit, [batch, -1, feature_dims])), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit + predicted_visit, :], predicted_trajectory))
            point_process_loss = -tf.reduce_mean(log_f_t_i_all)
            loss = mse_loss + point_process_loss * point_process_imbalance

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in point_process.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                m = encode_share.get_weights()[0]
                logged.add(train_set.epoch_completed)
                loss_pre = loss
                mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit + predicted_visit, :], predicted_trajectory))
                point_process_loss = -tf.reduce_mean(log_f_t_i_all)
                loss = mse_loss + point_process_loss * point_process_imbalance
                loss_diff = loss_pre - loss

                if loss > max_loss:
                    count = 0
                else:
                    if loss_diff > max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

                input_x_test = test_set[:, :, 1:]
                batch_test = tf.shape(input_x_test)[0]
                input_t_test = tf.reshape(test_set[:, :, 0], [batch_test, -1, 1])
                max_value = tf.reduce_max(input_t_test)
                input_t_test = input_t_test
                # TODO: 注释
                # one_year_time = np.load('../../Trajectory_generate/resource/HF_1_year__time.npy').reshape(batch_test, -1,1)
                # two_year_time = np.load('../../Trajectory_generate/resource/HF_2_year_time.npy').reshape(batch_test, -1,1)
                # three_month_time = np.load('../../Trajectory_generate/resource/HF_3_month_time.npy')

                predicted_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                predicted_trajectory_test_one_year = tf.zeros(shape=[batch_test, 0, feature_dims])
                predicted_trajectory_test_two_year = tf.zeros(shape=[batch_test, 0, feature_dims])

                log_f_t_i_all_test = tf.zeros(shape=[batch_test, 1])
                for predicted_visit_ in range(predicted_visit):
                    for previous_visit_ in range(previous_visit):
                        feature_test = input_x_test[:, previous_visit_, :]
                        time_test = tf.reshape(input_t_test[:, previous_visit_, :], [batch_test, -1])
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                        encode_c_test, encode_h_test = encode_share([feature_test, time_test, encode_c_test, encode_h_test])
                        encode_c_test_one_year = encode_c_test
                        encode_h_test_one_year = encode_h_test

                        encode_c_test_two_year = encode_c_test
                        encode_h_test_two_year = encode_h_test

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            time_test = input_t_test[:, previous_visit+i, :]
                            # TODO: 注释
                            # time_test_one_year = one_year_time[:, previous_visit+i, :] - 85 * i
                            # time_test_two_year = two_year_time[:, previous_visit+i, :]

                            encode_c_test, encode_h_test = encode_share([predicted_trajectory_test[:, i, :], time_test, encode_c_test, encode_h_test])
                            # todo: 注释
                            # encode_c_test_one_year, encode_h_test_one_year = encode_share([predicted_trajectory_test_one_year[:, i, :], time_test_one_year, encode_c_test_one_year, encode_h_test_one_year])
                            # encode_c_test_two_year, encode_h_test_two_year = encode_share([predicted_trajectory_test_two_year[:, i, :], time_test_two_year, encode_c_test_two_year, encode_h_test_two_year])

                    if predicted_visit_ == 0:
                        decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                        # todo: 注释
                        # decode_c_test_one_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        # decode_h_test_one_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        #
                        # decode_c_test_two_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        # decode_h_test_two_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                    generated_next_visit_test, decode_c_test, decode_h_test = decoder_share([encode_h_test, decode_c_test, decode_h_test])
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(generated_next_visit_test, [batch_test, -1, feature_dims])), axis=1)

                    # TODO: 注释
                    # generated_next_visit_test_one_year, decode_c_test_one_year, decode_h_test_one_year = decoder_share([encode_h_test_one_year, decode_c_test_one_year, decode_h_test_one_year])
                    # predicted_trajectory_test_one_year = tf.concat((predicted_trajectory_test_one_year, tf.reshape(generated_next_visit_test_one_year, [batch_test, -1, feature_dims])), axis=1)
                    #
                    # generated_next_visit_test_two_year, decode_c_test_two_year, decode_h_test_two_year = decoder_share([encode_h_test_two_year, decode_c_test_two_year, decode_h_test_two_year])
                    # predicted_trajectory_test_two_year = tf.concat((predicted_trajectory_test_two_year, tf.reshape(generated_next_visit_test_two_year, [batch_test, -1, feature_dims])), axis=1)

                    time_interval_test = input_t_test[:, previous_visit+predicted_visit_, :] - input_t_test[:, previous_visit+predicted_visit_-1, :]
                    log_f_t_i_test = point_process([encode_h_test, time_interval_test])
                    log_f_t_i_all_test = tf.concat((log_f_t_i_all_test, tf.reshape(log_f_t_i_test, [batch_test, -1])), axis=1)

                mse_loss_predicted = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test))
                mae_predicted = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test))

                # todo: 注释
                # mse_loss_predicted_one_year = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test_one_year))
                # mae_predicted_one_year = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test_one_year))
                #
                # mse_loss_predicted_two_year = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test_two_year))
                # mae_predicted_two_year = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :], predicted_trajectory_test_two_year))

                point_process_loss_test = -tf.reduce_mean(log_f_t_i_test)
                r_value_all = []
                p_value_all = []
                r_value_all_one_year = []
                r_value_all_two_year = []
                # for r in range(predicted_visit):
                #     x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                #     y_ = tf.reshape(predicted_trajectory_test[:, r, :], (-1,))
                #     z_ = tf.reshape(predicted_trajectory_test_one_year[:, r, :], (-1,))
                #     h_ = tf.reshape(predicted_trajectory_test_two_year[:, r, :], (-1,))
                #     if (y_.numpy() == np.zeros_like(y_)).all():
                #         r_value_ = [0, 0]
                #     else:
                #         if (z_.numpy() == np.zeros_like(y_)).all():
                #             r_value_one_year = [0, 0]
                #         else:
                #             if (h_.numpy() == np.zeros_like(y_)).all():
                #                 r_value_two_year = [0, 0]
                #
                #             else:
                #                 r_value_ = stats.pearsonr(x_, y_)
                #                 r_value_one_year = stats.pearsonr(x_, z_)
                #                 r_value_two_year = stats.pearsonr(x_, h_)
                #     r_value_all.append(r_value_[0])
                #     r_value_all_one_year.append(r_value_one_year[0])
                #     r_value_all_two_year.append(r_value_two_year[0])
                #     p_value_all.append(r_value_[1])

                for r in range(predicted_visit-1):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(predicted_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                print('--epoch{}------mse_loss{}----predicted_mse----mae_predicted---{}-'
                      '{}---predicted_r_value---{}--predicted_point_process_loss---{}'
                      'count  {}'.format(train_set.epoch_completed,
                                         mse_loss, mse_loss_predicted,
                                         mae_predicted,
                                         np.mean(r_value_all),
                                         point_process_loss_test,
                                         count))

                # print('epoch---{}---mse_real-{}---mse_one_year--{}---mse_two_year---{}--'
                #       'mae_real---{}---mae_one_year---{}---mae_two_year---{}---'
                #       'r_value_real---{}---mae_one_year---{}---r_value_two_year---{}'
                #       .format(train_set.epoch_completed,
                #               mse_loss_predicted,
                #               mse_loss_predicted_one_year,
                #               mse_loss_predicted_two_year,
                #               mae_predicted,
                #               mae_predicted_one_year,
                #               mae_predicted_two_year,
                #               np.mean(r_value_all),
                #               np.mean(r_value_all_one_year),
                #               np.mean(r_value_all_two_year)
                #               ))

        tf.compat.v1.reset_default_graph()
    return mse_loss_predicted, mae_predicted, np.mean(r_value_all)
    # return -1*np.mean(mse_loss_predicted)


if __name__ == '__main__':
    test_test('SS_RMTPP_HF_test_3_3.txt')
    # SS_RMTPP_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #         'point_process_imbalance': (-3, 2)
    #     }
    # )
    # SS_RMTPP_BO.maximize()
    # print(SS_RMTPP_BO.max)

    mse_all = []
    r_value_all = []
    mae_all = []
    for i in range(50):
        mse, mae, r_value = train(hidden_size=64,
                                  learning_rate=0.0006090511300247582,
                                  l2_regularization=0.00014726875572420634,
                                  point_process_imbalance=21.117106101749297)
        mse_all.append(mse)
        r_value_all.append(r_value)
        mae_all.append(mae)
        print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
              "r_value_std {}----mse_all_std  {}  mae_std {}".
              format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
                     np.std(r_value_all), np.std(mse_all),np.std(mae_all)))







