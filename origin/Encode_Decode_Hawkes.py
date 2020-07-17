import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from modify.data import DataSet
from LSTMCell import *


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
        shape_weight_1 = tf.TensorShape((1, 1, context_state_shape[0][2]))
        shape_weight_2 = tf.TensorShape((1, 1, context_state_shape[0][2]))
        shape_weight_3 = tf.TensorShape((1, 1, context_state_shape[0][2]))
        shape_weight_hawkes = tf.TensorShape((1, 1))

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

        self.trigger_parameter_beta = self.add_weight(name='trigger_parameter_beta',
                                                      initializer='uniform',
                                                      shape=shape_weight_hawkes,
                                                      trainable=True)
        self.trigger_parameter_alpha = self.add_weight(name='trigger_parameter_alpha',
                                                       initializer='uniform',
                                                       shape=shape_weight_hawkes,
                                                       trainable=True)
        self.base_intensity = self.add_weight(name='base_intensity',
                                              initializer='uniform',
                                              shape=shape_weight_hawkes,
                                              trainable=True)

        super(Decode, self).build(context_state_shape)

    def call(self, context_state):  # 这里考虑了时间因素
        context_state, input_t = context_state
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
            input_t_ = tf.reshape(input_t, [batch, self.time_step])
            current_time_index = time+3
            output, state = self.LSTM_Cell_decode(decoder_input[:, time, :], state)
            c = state[0]
            h = state[1]
            trigger_paramaters_beta = tf.tile(self.trigger_parameter_beta, [batch, 1])
            trigger_paramaters_alpha = tf.tile(self.trigger_parameter_alpha, [batch, 1])
            base_intensity = tf.tile(self.base_intensity, [batch, 1])
            condition_intensity_value = self.calculate_hawkes_process(batch=batch, input_t=input_t_,
                                                                      current_time_index=current_time_index,
                                                                      trigger_parameter_alpha=trigger_paramaters_alpha,
                                                                      trigger_parameter_beta=trigger_paramaters_beta,
                                                                      base_intensity=base_intensity)
            condition_intensity_value_ = tf.tile(tf.reshape(condition_intensity_value, [batch, 1]), [1, self.feature_dims])
            h = h*condition_intensity_value_
            state = [c, h]
            output_ = tf.reshape(output, [batch, 1, -1])
            fake_input = tf.concat((fake_input, output_), axis=1)
        fake_input = tf.reshape(fake_input, [-1, self.feature_dims])
        fake_input = self.dense1(fake_input)
        fake_input = self.dense2(fake_input)
        return tf.reshape(self.dense3(fake_input), (-1, 3, self.feature_dims))

    def calculate_hawkes_process(self, batch, input_t, current_time_index,
                                 trigger_parameter_beta, trigger_parameter_alpha, base_intensity):

        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])  # number value[batch ,1]
        current_t_tile = tf.tile(current_t, [1, current_time_index])  # [batch, current_index]

        time_before_t = input_t[:, :current_time_index]
        time_difference = time_before_t - current_t_tile

        triggering_kernel = tf.reduce_sum(tf.exp(time_difference) * trigger_parameter_beta, axis=1)

        triggering_kernel = tf.reshape(triggering_kernel, [batch, 1])
        condition_intensity_value = base_intensity + trigger_parameter_alpha*triggering_kernel
        return condition_intensity_value


def train_step(hidden_size, lambda_balance, learning_rate, l2_regularization):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)
    # # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)

    # train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)
    # # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)
    # test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)

    train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)
    # test_set = np.load('HF_validate_.npy').reshape(-1, 6, 30)

    time_step = 6

    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0

    batch_size = 64
    epochs = 1
    #
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
            input_x_train_feature = tf.cast(input_x_train[:, :, 1:], tf.float32)
            input_t_train = input_x_train[:, :, 0].reshape(-1, 6, 1)
            input_t_train = tf.cast(input_t_train, tf.float32)

            input_x_test_feature = test_set.dynamic_features
            input_x_test = tf.cast(input_x_test_feature[:, :, 1:], tf.float32)
            input_t_test = input_x_test_feature[:, :, 0].reshape(-1, 6, 1)
            input_t_test = tf.cast(input_t_test, tf.float32)

            context_state = encode_context(input_x_train_feature)
            fake_input = decode([context_state, input_t_train])

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
        fake_input_test = decode([context_state_test, input_t_test])
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

    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=128,  lambda_balance=0.00051054, learning_rate=0.01045, l2_regularization=0.00001049)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=64,  lambda_balance=1.0, learning_rate=0.01, l2_regularization=0.0011137686)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    mse_all = []
    for i in range(50):
        mse = train_step(hidden_size=128,  lambda_balance=0.0000047594, learning_rate=0.03198425115068335, l2_regularization=0.00001878718618)
        mse_all.append(mse)
        print('第{}次测试完成'.format(i))
    print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    print('----------------mse_std:{}----------'.format(np.std(mse_all)))



























