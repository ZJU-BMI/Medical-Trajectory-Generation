import tensorflow as tf
from tensorflow.keras.models import Model
from data import DataSet
from bayes_opt import BayesianOptimization
from TimeLSTMCell_1 import *
import scipy.stats as stats
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def DynamicTimeWarping(ptSetA, ptSetB):
    # 获得点集ptSetA中点的个数n
    n = ptSetA.shape[0]
    # 获得点集ptSetB中点的个数m
    m = ptSetB.shape[0]
    # 计算任意两个点的距离矩阵
    disMat = cdist(ptSetA, ptSetB, metric='euclidean')
    # 初始化消耗矩阵
    costMatrix = np.full((n,m),-1.0)
    # 递归求解DTW距离
    dtwDis = _dtw(disMat,costMatrix,n-1,m-1)
    return dtwDis


def _dtw(disMat,costMatrix,i,j):
    # 如果costMatrix[i][j]不等于-1，直接返回，不需要计算了（借助动态规划的思想）
    if costMatrix[i][j] > -1:
        return costMatrix[i][j]
    # 当i,j都等于0的时候，计算消耗矩阵的值
    if i == 0 and j == 0:
        costMatrix[i][j] = disMat[0][0]
    # 计算第一列的值
    if i > 0 and j == 0:
        costMatrix[i][j] = _dtw(disMat, costMatrix, i - 1, 0) + disMat[i][0]
    # 计算第一行的值
    if i == 0 and j > 0:
        costMatrix[i][j] = _dtw(disMat, costMatrix, 0, j - 1) + disMat[0][j]
    # 计算其他值
    if i > 0 and j > 0:
        costMatrix[i][j] = min(_dtw(disMat, costMatrix, i, j-1),
                               _dtw(disMat, costMatrix, i - 1, j-1),
                               _dtw(disMat, costMatrix, i - 1, j)) + disMat[i][j]
    return costMatrix[i][j]


# 单步x into hidden representation
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


class Decoder(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_share')
        self.feature_dims = feature_dims
        self.hidden_size = hidden_size
        self.LSTM_decoder = TimeLSTMCell_1(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, input_x):
        sequence_time, input_t, c, h = input_x
        state = [c, h]
        output, state = self.LSTM_decoder([sequence_time, input_t], state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]


def train(hidden_size, learning_rate, l2_regularization):

    # train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)

    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

    # sepsis mimic dataset
    train_set = np.load('../../Trajectory_generate/dataset_file/sepsis_mimic_train.npy').reshape(-1, 13, 40)
    test_set = np.load('../../Trajectory_generate/dataset_file/sepsis_mimic_test.npy').reshape(-1, 13, 40)
    # test_set = np.load('../../Trajectory_generate/dataset_file/sepsis_mimic_validate.npy').reshape(-1, 13, 40)

    previous_visit = 3
    predicted_visit = 10

    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    train_set.epoch_completed = 0
    batch_size = 64

    epochs = 50

    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))

    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate, l2_regularization))

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)

    logged = set()

    max_loss = 0.01
    max_pace = 0.0001

    mse_loss = 0
    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size)
        input_x_train = input_train[:, :, 1:]
        input_t_train = input_train[:, :, 0]
        batch = input_x_train.shape[0]
        with tf.GradientTape() as tape:
            predicted_trajectory = np.zeros(shape=(batch, 0, feature_dims))
            for predicted_visit_ in range(predicted_visit):
                sequence_time_last_time = input_x_train[:, previous_visit+predicted_visit_-1, :]  # y_j
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h  # h_j from 1 to j

                input_decode = tf.concat((sequence_time_last_time, context_state), axis=1)  # y_j and h_j
                if predicted_visit_ == 0:
                    decode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                input_t = tf.reshape(input_t_train[:, previous_visit+predicted_visit_], [-1, 1])
                predicted_next_sequence, decode_c, decode_h = decoder_share([input_decode, input_t, decode_c, decode_h])
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit+predicted_visit, :], predicted_trajectory))

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                mse_loss += tf.keras. regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                loss_pre = mse_loss
                mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit + predicted_visit, :], predicted_trajectory))
                loss_diff = loss_pre - mse_loss
                if mse_loss > max_loss:
                    count = 0

                else:
                    if loss_diff > max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

                input_x_test = test_set[:, :, 1:]
                input_t_test = test_set[:, :, 0]
                batch_test = input_x_test.shape[0]
                predicted_trajectory_test = np.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    if predicted_visit_ == 0:
                        sequence_time_last_time_test = input_x_test[:, predicted_visit_+previous_visit-1, :]
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            sequence_input_t = predicted_trajectory_test[:, i, :]
                            encode_c_test, encode_h_test = encode_share([sequence_input_t, encode_c_test, encode_h_test])

                    context_state = encode_h_test

                    if predicted_visit_ == 0:
                        decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    input_decode_test = tf.concat((sequence_time_last_time_test, context_state), axis=1)
                    input_t = tf.reshape(input_t_test[:, previous_visit+predicted_visit_], [-1, 1])
                    predicted_next_sequence_test, decode_c_test, decode_h_test = decoder_share([input_decode_test, input_t, decode_c_test, decode_h_test])
                    sequence_time_last_time_test = predicted_next_sequence_test  # feed the generated sequence into next state
                    predicted_next_sequence_test = tf.reshape(predicted_next_sequence_test, [batch_test, -1, feature_dims])
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, predicted_next_sequence_test),
                                                          axis=1)
                mse_loss_predicted = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
                mae_predicted = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))

                r_value_all = []
                for patient in range(batch_test):
                    r_value = 0.0
                    for feature in range(feature_dims):
                        x_ = input_x_test[patient, previous_visit:, feature].reshape(predicted_visit, 1)
                        y_ = predicted_trajectory_test[patient, :, feature].numpy().reshape(predicted_visit, 1)
                        r_value += DynamicTimeWarping(x_, y_)
                    r_value_all.append(r_value / 29.0)
                print('------epoch{}------mse_loss{}----predicted_mse-----{}---predicted_r_value---{}-'
                      '-count  {}'.format(train_set.epoch_completed, mse_loss,
                                          mse_loss_predicted, np.mean(r_value_all), count))

                # r_value_all = []
                # p_value_all = []
                # r_value_spearman = []
                # r_value_kendalltau = []
                # for visit in range(predicted_visit):
                #     for feature in range(feature_dims):
                #         x_ = input_x_test[:, previous_visit+visit, feature]
                #         y_ = predicted_trajectory_test[:, visit, feature]
                #         r_value_ = stats.pearsonr(x_, y_)
                #         r_value_spearman_ = stats.spearmanr(x_, y_)
                #         r_value_kendalltau_ = stats.kendalltau(x_, y_)
                #         if not np.isnan(r_value_[0]):
                #             r_value_all.append(np.abs(r_value_[0]))
                #             p_value_all.append(np.abs(r_value_[1]))
                #         if not np.isnan(r_value_spearman_[0]):
                #             r_value_spearman.append(np.abs(r_value_spearman_[0]))
                #         if not np.isnan(r_value_kendalltau_[0]):
                #             r_value_kendalltau.append(np.abs(r_value_kendalltau_[0]))
                # print('------epoch{}------mse_loss{}----predicted_mse-----{}---predicted_r_value---{}-'
                #       'predicted_spearman---{}-'
                #       '--predicted_kendalltau---{}-count  {}'.format(train_set.epoch_completed, mse_loss,
                #                                                      mse_loss_predicted, np.mean(r_value_all),
                #                                                      np.mean(r_value_spearman),
                #                                                      np.mean(r_value_kendalltau), count))
    tf.compat.v1.reset_default_graph()
    return mse_loss_predicted, mae_predicted, np.mean(r_value_all), np.mean(p_value_all)
    # return mse_loss_predicted, mae_predicted, np.mean(r_value_all), np.mean(p_value_all), np.mean(r_value_spearman), np.mean(r_value_kendalltau)
    # return -1*mse_loss_predicted


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
    test_test('AED_Time_1—sepsis_test_DTW.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    mse_all = []
    mae_all = []
    r_value_all = []
    p_value_all = []
    r_value_spearman_all = []
    r_value_kendalltau_all = []
    for i in range(50):
        mse, mae, r_value, p_value = \
            train(hidden_size=128,
                  learning_rate=0.0004762417557512286,
                  l2_regularization=1.0804187569628365e-05)
        # mse, mae, r_value, p_value, r_value_spearman, r_value_kendalltau = \
        #      train(hidden_size=32,
        #            learning_rate=0.014320094531412003,
        #            l2_regularization=2.7724680953816745e-05)
        mse_all.append(mse)
        r_value_all.append(r_value)
        # r_value_kendalltau_all.append(r_value_kendalltau)
        # r_value_spearman_all.append(r_value_spearman)
        p_value_all.append(p_value)
        mae_all.append(mae)
        print('epoch  {}-----mse-all_ave  {}----mae_all_ave-----{}---r_value_ave  {}--'
              '---p_value_ave  {}--  mse_vale_std{}------mae_vale_std{}---r_value_std{}  p_value_std{}-'.
              format(i, np.mean(mse_all), np.mean(mae_all),
                     np.mean(r_value_all), np.mean(p_value_all),
                     np.std(mse_all), np.std(mae_all),
                     np.std(r_value_all), np.std(p_value_all)))

