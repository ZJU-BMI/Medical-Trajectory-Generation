import tensorflow as tf
import numpy as np
from tensorflow_core.python.keras.models import Model
import os
from data import DataSet
from utils import Decoder, test_test, Encoder
from scipy import stats
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings(action='once')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
        trigger_alpha = tf.tile(tf.keras.activations.sigmoid(self.trigger_parameter_alpha), [batch, 1])
        trigger_beta = tf.tile(tf.keras.activations.sigmoid(self.trigger_parameter_beta), [batch, 1])
        base_intensity = tf.tile(tf.keras.activations.sigmoid(self.base_intensity), [batch, 1])

        condition_intensity = self.calculate_lambda_process(input_t, current_time_index,
                                                            trigger_alpha, trigger_beta, base_intensity)
        likelihood = self.calculate_likelihood(input_t, current_time_index, trigger_alpha,
                                               trigger_beta, base_intensity)
        return condition_intensity, likelihood


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


def train(hidden_size, l2_regularization, learning_rate, generated_imbalance, likelihood_imbalance):
    train_set = np.load("../../Trajectory_generate/dataset_file/HF_train_.npy").reshape(-1, 6, 30)
    # test_set = np.load("../../Trajectory_generate/dataset_file/HF_test_.npy").reshape(-1, 6, 30)
    test_set = np.load("../../Trajectory_generate/dataset_file/HF_validate_.npy").reshape(-1, 6, 30)

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50

    hidden_size = 2 ** (int(hidden_size))
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    generated_imbalance = 10 ** generated_imbalance
    likelihood_imbalance = 10 ** likelihood_imbalance

    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))

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

                generated_next_visit, decode_c, decode_h = decoder_share([context_state*condition_intensity, decode_c, decode_h])
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
                batch_test = input_x_test.shape[0]
                input_t_test = tf.cast(test_set[:, :, 0], tf.float32)
                # one_year_time = np.load('../../Trajectory_generate/resource/HF_1_year__time.npy').reshape(batch_test, -1)

                predicted_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                predicted_trajectory_test_one_year = tf.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])
                    encode_c_test_one_year, encode_h_test_one_year = encode_c_test, encode_h_test

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            encode_c_test, encode_h_test = encode_share([predicted_trajectory_test[:, i, :], encode_c_test, encode_h_test])
                            # encode_c_test_one_year, encode_h_test_one_year = encode_share([predicted_trajectory_test_one_year[:, i, :], encode_c_test_one_year, encode_h_test_one_year])

                    context_state_test = encode_h_test
                    context_state_test_one_year = encode_h_test_one_year

                    if predicted_visit_ == 0:
                        decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                        decode_c_test_one_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test_one_year = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                    current_time_index_shape_test = tf.ones(shape=[previous_visit+predicted_visit_])
                    condition_intensity_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])
                    sequence_next_visit_test, decode_c_test, decode_h_test = decoder_share([context_state_test*condition_intensity_test, decode_c_test, decode_h_test])
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(sequence_next_visit_test, [batch_test, -1, feature_dims])), axis=1)

                    # condition_intensity_test_one_year, likelihood_test = hawkes_process([one_year_time, current_time_index_shape_test])
                    # sequence_next_visit_test_one_year, decode_c_test_one_year, decode_h_test_one_year = decoder_share([context_state_test_one_year*condition_intensity_test_one_year, decode_c_test_one_year, decode_h_test_one_year])
                    # predicted_trajectory_test_one_year = tf.concat((predicted_trajectory_test_one_year, tf.reshape(sequence_next_visit_test_one_year, [batch_test, -1, feature_dims])), axis=1)

                mse_generated_loss_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
                mae_generated_loss_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))

                # mse_generated_loss_test_one_year = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test_one_year))
                # mae_generated_loss_test_one_year = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit + predicted_visit, :],predicted_trajectory_test_one_year))

                r_value_all = []
                p_value_all = []
                r_value_all_one_year = []

                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(predicted_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)

                    # z_ = tf.reshape(predicted_trajectory_test_one_year[:, r, :], (-1,))
                    # r_value_one_year_ = stats.pearsonr(x_, z_)
                    #
                    # r_value_all_one_year.append(r_value_one_year_)

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

                # print("epoch  {}---train_mse_generate {}- - "
                #       "mae_generated_loss--{}--test_mse {}---{}-test_mae--{}  "
                #       "--{}---r_value {}  -{}----count {}-".format(train_set.epoch_completed,
                #                                                    mse_generated_loss,
                #                                                    mae_generated_loss,
                #                                                    mse_generated_loss_test,
                #                                                    mse_generated_loss_test_one_year,
                #                                                    mae_generated_loss_test,
                #                                                    mae_generated_loss_test_one_year,
                #                                                    np.mean(r_value_all),
                #                                                    np.mean(r_value_all_one_year),
                #                                                    count))
    tf.compat.v1.reset_default_graph()
    # return mse_generated_loss_test, mae_generated_loss_test, np.mean(r_value_all)
    return -1 * mse_generated_loss_test


if __name__ == '__main__':
    test_test('AED_Hawkes_HF_train_3_3_时间参数大于0.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
        train, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, 1),
            'l2_regularization': (-5, 1),
            'generated_imbalance': (-6, 1),
            'likelihood_imbalance': (-6, 1)
        }
    )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)

    # mse_all = []
    # r_value_all = []
    # mae_all = []
    # for i in range(50):
    #     mse, mae, r_value = train(hidden_size=256,
    #                               learning_rate=0.0022414493262367242,
    #                               l2_regularization=1e-5,
    #                               generated_imbalance=10.0,
    #                               likelihood_imbalance=0.5543000039984665)
    #     mse_all.append(mse)
    #     r_value_all.append(r_value)
    #     mae_all.append(mae)
    #     print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
    #           "r_value_std {}----mse_all_std  {}  mae_std {}".
    #           format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
    #                  np.std(r_value_all), np.std(mse_all),np.std(mae_all)))








