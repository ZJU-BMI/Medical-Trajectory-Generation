import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet, read_data, read_gaucoma_data
import numpy as np
from datetime import datetime
from tensorflow_core.python.keras import backend as K
from sklearn.model_selection import train_test_split
from LSTMCell import *
from bayes_opt import BayesianOptimization


# Define encode class: encode the history information to deep representation[batch_size, 3, hidden_size]
class EncodeContext(Model):
    def __init__(self, hidden_size, time_step, batch_size):
        super().__init__(name='encode_context')
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.batch_size = batch_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_context):
        batch = tf.shape(input_context)[0]
        state = self.LSTM_Cell_encode.get_initial_state(batch_size=batch, dtype=tf.float32)
        context_state = np.zeros(shape=[batch, 0, self.hidden_size])
        for time in range(self.time_step-3):
            input = input_context[:, time, :]
            output, state = self.LSTM_Cell_encode(input, state)
            output = tf.reshape(output, [batch, 1, -1])
            context_state = np.concatenate((context_state, output), axis=1)
        return context_state


# Decode and obtain the next few steps information according the deep representation of history information
class Decode(Model):
    def __init__(self, feature_dims, hidden_size, batch_size, time_step):
        super().__init__(name='generator')
        self.feature_dims = feature_dims
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.time_step = time_step
        self.LSTM_Cell_decode = LSTMCell(feature_dims)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def build(self, context_state_shape):
        shape_weight_1 = tf.TensorShape((1, 1, context_state_shape[2]))
        shape_weight_2 = tf.TensorShape((1, 1, context_state_shape[2]))
        shape_weight_3 = tf.TensorShape((1, 1, context_state_shape[2]))

        self.attention_weight_1 = self.add_weight(name='attention_weight_1',
                                                  shape=shape_weight_1,
                                                  initializer='uniform',
                                                  trainable=True)
        self.attention_weight_2 = self.add_weight(name='attention_weight_2',
                                                  shape=shape_weight_2,
                                                  initializer='uniform',
                                                  trainable=True)
        self.attention_weight_3 = self.add_weight(name='attention_weight_3',
                                                  initializer='uniform',
                                                  shape=shape_weight_3,
                                                  trainable=True)

        super(Decode, self).build(context_state_shape)

    def call(self, context_state):  # 这里考虑了时间因素
        batch = tf.shape(context_state)[0]
        last_hidden = context_state[:, 2, :]
        last_hidden = tf.reshape(last_hidden, (-1, 1, self.hidden_size))
        last_hidden_all = tf.tile(last_hidden, [1, self.time_step - 3, 1])
        attention_weight_1 = tf.tile(self.attention_weight_1, [batch, 1, 1])
        attention_weight_2 = tf.tile(self.attention_weight_2, [batch, 1, 1])
        attention_weight_3 = tf.tile(self.attention_weight_3, [batch, 1, 1])
        attention_weight = tf.concat((tf.concat((attention_weight_1, attention_weight_2), axis=1),
                                      attention_weight_3), axis=1)
        decoder_input = last_hidden_all * attention_weight  # (batch_size, self.time_step-3, hidden_size)
        # state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=tf.float32)
        state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=float)
        fake_input = tf.zeros(shape=[batch, 0, self.feature_dims])
        for time in range(self.time_step - 3):
            output, state = self.LSTM_Cell_decode(decoder_input[:, time, :], state)
            output_ = tf.reshape(output, [batch, 1, -1])
            fake_input = tf.concat((fake_input, output_), axis=1)
        fake_input = tf.reshape(fake_input, [-1, self.feature_dims])
        fake_input = self.dense1(fake_input)
        fake_input = self.dense2(fake_input)
        return tf.reshape(self.dense3(fake_input), (-1, 3, self.feature_dims))


def train_step(hidden_size, lambda_balance, learning_rate, l2_regularization):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)[:, :, 1:]

    train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)[:, :, 1:]
    # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)[:, :, 1:]
    test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)[:, :, 1:]


    time_step = 6

    feature_dims = train_set.shape[2]

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0

    batch_size = 64
    epochs = 1

    # hidden_size = 2 ** (int(hidden_size))
    # lambda_balance = 10 ** lambda_balance
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('----batch_size{}---hidden_size{}-----epochs{}---'
          'lambda_balance{}---learning_rate{}---l2_regularization{}---'
          .format(batch_size, hidden_size, epochs, lambda_balance, learning_rate, l2_regularization))

    decode = Decode(feature_dims=feature_dims,
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    time_step=time_step)

    encode_context = EncodeContext(hidden_size=hidden_size,
                                   time_step=time_step,
                                   batch_size=batch_size)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    with tf.GradientTape(persistent=True) as tape:
        while train_set.epoch_completed < epochs:
            input_x_train = train_set.next_batch(batch_size)
            input_x_train_feature = tf.cast(input_x_train, tf.float32)

            input_x_test = test_set.dynamic_features
            input_x_test = tf.cast(input_x_test, tf.float32)

            context_state = encode_context(input_x_train_feature)
            fake_input = decode(context_state)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], fake_input))

            for weight in decode.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in encode_context.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = [var for var in decode.trainable_variables]
            for var in encode_context.trainable_variables:
                variables.append(var)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

        context_state_test = encode_context(input_x_test)
        fake_input_test = decode(context_state_test)
        input_r = input_x_test[:, 3:, :]
        input_f = fake_input_test
        mse_loss_ = tf.reduce_mean(tf.keras.losses.MSE(input_r, input_f))

        print("-------mse_loss:{}---------".format(mse_loss_))
        tf.compat.v1.reset_default_graph()
        return -1 * mse_loss_


if __name__ == '__main__':
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train_step, {
    #         'hidden_size': (5, 8),
    #         'lambda_balance': (-6, 0),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    mse_all = []
    for i in range(50):
        mse = train_step(hidden_size=64,  lambda_balance=0.28497, learning_rate=0.068058, l2_regularization=0.0006203)
        mse_all.append(mse)
        print('第{}次测试完成'.format(i))
    print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    print('----------------mse_std:{}----------'.format(np.std(mse_all)))





























