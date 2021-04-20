import tensorflow as tf
from tensorflow.keras.models import Model
from data import DataSet
from bayes_opt import BayesianOptimization
import scipy.stats as stats
import numpy as np
import sys
import os
from TimeLSTMCell_3 import *
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings(action='ignore')

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


# 单步执行编码的过程
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


# 输入的数据是z_(i+1), h_i, x_i 输出x_(i+1)
class Decoder(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_share')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.LSTM_Cell_decode = TimeLSTMCell_3(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, input_x):
        z_i_1, h_i, x_i, decode_c, decode_h, input_t = input_x
        input_decode = tf.concat((z_i_1, h_i, x_i), axis=1)
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decode([input_decode, input_t], state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]


# 先验网络： 输入是h_i 输出z_(i+1)
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
        # batch = tf.shape(mu)[0]
        # sample_all = tf.zeros(shape=(batch, 0))
        # for feature in range(z_dims):
        #     sample = tf.compat.v1.random_normal(shape=(batch, 1))
        #     sample_all = tf.concat((sample_all, sample), axis=1)
        # z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        sample_all = tf.compat.v1.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        return z


# 后验网络，输入h_i, h_(i+1) 输出z_(i+1)
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

    def reparameterize(self, mu, log_var, z_dims):
        # batch = tf.shape(mu)[0]
        # sample_all = tf.zeros(shape=(batch, 0))
        # for feature in range(z_dims):
        #     sample = tf.compat.v1.random_normal(shape=(batch, 1))
        #     sample_all = tf.concat((sample_all, sample), axis=1)
        # z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        sample_all = tf.compat.v1.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        z = mu + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var)))
        return z

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


def kl_loss(z_mean_post, log_var_post, z_mean_prior, log_var_prior):
    std_post = tf.math.sqrt(tf.exp(log_var_post))
    std_prior = tf.math.sqrt(tf.exp(log_var_prior))
    kl_loss_element = (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(tf.maximum(std_post, 1e-9)) +
                       (tf.math.pow(std_post, 2) +
                       tf.math.pow((z_mean_post - z_mean_prior), 2)) / tf.maximum(tf.math.pow(std_prior, 2), 1e-9)-1)
    return 0.5 * kl_loss_element


def train(hidden_size, z_dims, l2_regularization, learning_rate, kl_imbalance, reconstruction_imbalance, generated_mse_imbalance):
    # train_set = np.load("../../Trajectory_generate/dataset_file/HF_train_.npy").reshape(-1, 6, 30)
    # test_set = np.load("../../Trajectory_generate/dataset_file/HF_test_.npy").reshape(-1, 6, 30)
    # test_set = np.load("../../Trajectory_generate/dataset_file/HF_validate_.npy").reshape(-1, 6, 30)

    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)[:, :, 1:]

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

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
    #
    # hidden_size = 2 ** (int(hidden_size))
    # z_dims = 2 ** (int(z_dims))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # kl_imbalance = 10 ** kl_imbalance
    # reconstruction_imbalance = 10 ** reconstruction_imbalance
    # generated_mse_imbalance = 10 ** generated_mse_imbalance

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
    decode_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    prior_net = Prior(z_dims=z_dims)
    post_net = Post(z_dims=z_dims)

    logged = set()
    max_loss = 0.01
    max_pace = 0.0001
    loss = 0
    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = tf.cast(input_train[:, :, 1:], dtype=tf.float32)
        input_t_train = tf.cast(input_train[:, :, 0], tf.float32)
        batch = input_x_train.shape[0]

        with tf.GradientTape() as tape:
            generated_trajectory = np.zeros(shape=[batch, 0, feature_dims])
            construct_trajectory = np.zeros(shape=[batch, 0, feature_dims])
            z_log_var_post_all = np.zeros(shape=[batch, 0, z_dims])
            z_mean_post_all = np.zeros(shape=[batch, 0, z_dims])
            z_log_var_prior_all = np.zeros(shape=[batch, 0, z_dims])
            z_mean_prior_all = np.zeros(shape=[batch, 0, z_dims])

            for predicted_visit_ in range(predicted_visit):
                sequence_last_time = input_x_train[:, predicted_visit_+previous_visit-1, :]
                sequence_time_current_time = input_x_train[:, predicted_visit_+previous_visit, :]

                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h
                z_prior, z_mean_prior, z_log_var_prior = prior_net(context_state)  # h_i--> z_(i+1)
                encode_c, encode_h = encode_share([sequence_time_current_time, encode_c, encode_h])  # h_(i+1)
                z_post, z_mean_post, z_log_var_post = post_net([context_state, encode_h])  # h_i, h_(i+1) --> z_(i+1)
                if predicted_visit_ == 0:
                    decode_c_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_generate = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                    decode_c_reconstruct = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h_reconstruct = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                input_t = tf.reshape(input_t_train[:, previous_visit + predicted_visit_], [-1, 1])
                construct_next_visit, decode_c_reconstruct, decode_h_reconstruct = decode_share([z_post, context_state, sequence_last_time, decode_c_reconstruct, decode_h_reconstruct, input_t])
                construct_next_visit = tf.reshape(construct_next_visit, [batch, -1, feature_dims])
                construct_trajectory = tf.concat((construct_trajectory, construct_next_visit), axis=1)

                generated_next_visit, decode_c_generate, decode_h_generate = decode_share([z_prior, context_state, sequence_last_time, decode_c_generate, decode_h_generate, input_t])
                generated_next_visit = tf.reshape(generated_next_visit, (batch, -1, feature_dims))
                generated_trajectory = tf.concat((generated_trajectory, generated_next_visit), axis=1)

                z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)
                z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)
                z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)

            mse_reconstruction = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, :], construct_trajectory))
            mse_generate = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit,:], generated_trajectory))

            std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
            std_prior = tf.math.sqrt(tf.exp(z_mean_prior_all))
            kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(tf.maximum(std_post,
                                                                                                               1e-9)) +
                                     (tf.math.pow(std_post, 2) + tf.math.pow((z_mean_post_all - z_mean_prior_all), 2)) /
                                     tf.maximum(tf.math.pow(std_prior, 2), 1e-9) - 1)
            kl_loss_all = tf.reduce_mean(kl_loss_element)

            loss += mse_reconstruction * reconstruction_imbalance + kl_loss_all * kl_imbalance + mse_generate * generated_mse_imbalance

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in post_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in prior_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)
            tape.watch(variables)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                loss_pre = mse_generate
                mse_reconstruction = tf.reduce_mean(
                    tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                        construct_trajectory))
                mse_generate = tf.reduce_mean(
                    tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit + predicted_visit, :],
                                        generated_trajectory))
                kl_loss_all = tf.reduce_mean(kl_loss(z_mean_post=z_mean_post_all, z_mean_prior=z_mean_prior_all,
                                                     log_var_post=z_log_var_post_all, log_var_prior=z_log_var_prior_all))
                loss = mse_reconstruction + mse_generate + kl_loss_all
                loss_diff = loss_pre - mse_generate

                if mse_generate > max_loss:
                    count = 0  # max_loss = 0.01

                else:
                    if loss_diff > max_pace:  # max_pace = 0.0001
                        count = 0
                    else:
                        count += 1

                if count > 9:
                    break

                input_test = test_set
                input_x_test = tf.cast(input_test[:, :, 1:], dtype=tf.float32)
                input_t_test = tf.cast(input_test[:, :, 0], tf.float32)
                batch_test = input_x_test.shape[0]
                generated_trajectory_test = np.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):

                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=(batch_test, hidden_size)))
                            encode_h_test = tf.Variable(tf.zeros(shape=(batch_test, hidden_size)))

                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            sequence_input_t = generated_trajectory_test[:, i, :]
                            encode_c_test, encode_h_test = encode_share([sequence_input_t, encode_c_test, encode_h_test])

                    context_state_test = encode_h_test
                    z_prior_test, z_mean_prior_test, z_log_var_prior_test = prior_net(context_state_test)

                    if predicted_visit_ == 0:
                        decode_c_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        sequence_last_time_test = input_x_test[:, predicted_visit_ + previous_visit - 1, :]

                    input_t = tf.reshape(input_t_test[:, previous_visit + predicted_visit_], [-1, 1])
                    sequence_last_time_test, decode_c_generate_test, decode_h_generate_test = decode_share([z_prior_test, context_state_test, sequence_last_time_test, decode_c_generate_test, decode_h_generate_test, input_t])
                    generated_next_visit_test = sequence_last_time_test
                    generated_next_visit_test = tf.reshape(generated_next_visit_test, [batch_test, -1, feature_dims])
                    generated_trajectory_test = tf.concat((generated_trajectory_test, generated_next_visit_test), axis=1)

                mse_generate_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit,:], generated_trajectory_test))
                mae_generate_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit,:], generated_trajectory_test))

                r_value_all = []
                for patient in range(batch_test):
                    r_value = 0.0
                    for feature in range(feature_dims):
                        x_ = input_x_test[patient, previous_visit:, feature].numpy().reshape(predicted_visit, 1)
                        y_ = generated_trajectory_test[patient, :, feature].numpy().reshape(predicted_visit, 1)
                        r_value += DynamicTimeWarping(x_, y_)
                    r_value_all.append(r_value / 29.0)
                print("epoch  {}---train_mse_generate {}--train_reconstruct {}--train_kl "
                      "{}--test_mse {}--test_mae  {}---"
                      "-r_value {}---count {}".format(train_set.epoch_completed,
                                                      mse_generate,
                                                      mse_reconstruction,
                                                      kl_loss_all,
                                                      mse_generate_test,
                                                      mae_generate_test,
                                                      np.mean(r_value_all),
                                                      count))
                # r_value_all = []
                # p_value_all = []
                # r_value_spearman_all = []
                # r_value_kendalltau_all = []
                # for visit in range(predicted_visit):
                #     for feature in range(feature_dims):
                #         x_ = input_x_test[:, previous_visit+visit, feature]
                #         y_ = generated_trajectory_test[:, visit, feature]
                #         r_value_pearson = stats.pearsonr(x_, y_)
                #         r_value_spearman = stats.spearmanr(x_, y_)
                #         r_value_kendalltau = stats.kendalltau(x_, y_)
                #         if not np.isnan(r_value_pearson[0]):
                #             r_value_all.append(np.abs(r_value_pearson[0]))
                #             p_value_all.append(np.abs(r_value_pearson[1]))
                #
                #         if not np.isnan(r_value_spearman[0]):
                #             r_value_spearman_all.append(np.abs(r_value_spearman[0]))
                #
                #         if not np.isnan(r_value_kendalltau[0]):
                #             r_value_kendalltau_all.append(np.abs(r_value_kendalltau[0]))

                # print("epoch  {}---train_mse_generate {}--train_reconstruct {}--train_kl "
                #       "{}--test_mse {}--test_mae  {}----r_value {}---r_value_spearman---{}-"
                #       "r_value_kendalltau---{}---count {}".format(train_set.epoch_completed,
                #                                                   mse_generate,
                #                                                   mse_reconstruction,
                #                                                   kl_loss_all,
                #                                                   mse_generate_test,
                #                                                   mae_generate_test,
                #                                                   np.mean(r_value_all),
                #                                                   np.mean(r_value_spearman_all),
                #                                                   np.mean(r_value_kendalltau_all),
                #                                                   count))
    tf.compat.v1.reset_default_graph()
    return mse_generate_test, mae_generate_test, np.mean(r_value_all)
    # return mse_generate_test, mae_generate_test, np.mean(r_value_all), np.mean(r_value_spearman_all), np.mean(r_value_kendalltau_all)
    # return -1*mse_generate_test


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
    test_test('VAE_Time_3_sepsis_重新训练_DTW_test.txt')
    mse_all = []
    r_value_all = []
    mae_all = []
    for i in range(50):
        mse, mae, r_value = \
            train(hidden_size=128,
                  learning_rate=0.00012393005620833582,
                  l2_regularization=0.0003305321527447188,
                  z_dims=128,
                  kl_imbalance=0.3646916733901115,
                  generated_mse_imbalance=0.0003208980303305599,
                  reconstruction_imbalance=4.767885655091974)
        mse_all.append(mse)
        r_value_all.append(r_value)
        mae_all.append(mae)
        print("epoch{}----r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
              "r_value_std {}----mse_all_std  {}  mae_std {}".
              format(i, np.mean(r_value_all), np.mean(mse_all),
                     np.mean(mae_all), np.std(r_value_all),
                     np.std(mse_all), np.std(mae_all)))

