import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet, read_data, read_gaucoma_data
import numpy as np
from datetime import datetime
from tensorflow_core.python.keras import backend as K
from sklearn.model_selection import train_test_split
from LSTMCell import *


# define the discriminator class
class Discriminator(Model):
    def __init__(self, time_step, batch_size):
        super().__init__(name='discriminator')
        self.time_step = time_step
        self.batch_size = batch_size
        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, real_input, fake_input):
        batch = tf.shape(real_input)[0]
        input_same = real_input[:, :self.time_step - 3, :]
        trajectory_real = []
        trajectory_fake = []
        trajectory_real_predict = tf.zeros(shape=[batch, 0, 1])
        trajectory_fake_predict = tf.zeros(shape=[batch, 0, 1])
        for index in range(self.time_step - 3):
            next_real = real_input[:, index + 3, :]
            next_fake = fake_input[:, index, :]
            next_real = tf.reshape(next_real, [batch, 1, -1])
            next_fake = tf.reshape(next_fake, [batch, 1, -1])
            trajectory_step_real = tf.concat((input_same, next_real), axis=1)
            trajectory_step_fake = tf.concat((input_same, next_fake), axis=1)
            trajectory_real.append(trajectory_step_real)
            trajectory_fake.append(trajectory_step_fake)
            trajectory_step_real = self.dense1(trajectory_step_real)
            trajectory_step_real_predict = self.dense2(trajectory_step_real)
            trajectory_step_fake = self.dense1(trajectory_step_fake)
            trajectory_step_fake_predict = self.dense2(trajectory_step_fake)
            trajectory_step_real_predict = tf.reshape(trajectory_step_real_predict, [batch, -1, 1])
            trajectory_step_fake_predict = tf.reshape(trajectory_step_fake_predict, [batch, -1, 1])
            trajectory_real_predict = tf.concat((trajectory_real_predict, trajectory_step_real_predict), axis=1)
            trajectory_fake_predict = tf.concat((trajectory_fake_predict, trajectory_step_fake_predict), axis=1)
        return trajectory_real_predict, trajectory_fake_predict


# Define the generator class: get the output y according to last hidden state and attention mechanism
class Generator(Model):
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

        super(Generator, self).build(context_state_shape)

    def call(self, context_state):  # 这里还没有考虑时间因素
        batch = tf.shape(context_state)[0]
        last_hidden = context_state[:, 2, :]
        last_hidden = tf.reshape(last_hidden, (-1, 1, self.hidden_size))
        last_hidden_all = tf.tile(last_hidden, [1, self.time_step-3, 1])
        attention_weight_1 = tf.tile(self.attention_weight_1, [batch, 1, 1])
        attention_weight_2 = tf.tile(self.attention_weight_2, [batch, 1, 1])
        attention_weight_3 = tf.tile(self.attention_weight_3, [batch, 1, 1])
        attention_weight = tf.concat((tf.concat((attention_weight_1, attention_weight_2), axis=1),
                                      attention_weight_3), axis=1)
        decoder_input = last_hidden_all * attention_weight  # (batch_size, self.time_step-3, hidden_size)
        # state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=tf.float32)
        state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=float)
        fake_input = tf.zeros(shape=[batch, 0, self.feature_dims])
        for time in range(self.time_step-3):
            output, state = self.LSTM_Cell_decode(decoder_input[:, time, :], state)
            output_ = tf.reshape(output, [batch, 1, -1])
            fake_input = tf.concat((fake_input, output_), axis=1)
        fake_input = tf.reshape(fake_input, [-1, self.feature_dims])
        fake_input = self.dense1(fake_input)
        fake_input = self.dense2(fake_input)
        return tf.reshape(self.dense3(fake_input), (-1, 3, self.feature_dims))


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


def train_step(batch_size, hidden_size, time_step, feature_dims, n_disc, train_set,
               test_set, epochs, lambda_balance, learning_rate, l2_regularization):
    discriminator = Discriminator(time_step=time_step, batch_size=batch_size)
    generator = Generator(feature_dims=feature_dims,
                          hidden_size=hidden_size,
                          batch_size=batch_size,
                          time_step=time_step)

    encode_context = EncodeContext(hidden_size=hidden_size,
                                   time_step=time_step,
                                   batch_size=batch_size)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape, tf.GradientTape() as encode_tape:
        for epoch in range(epochs):
            input_x_train = train_set.next_batch(batch_size)
            input_x_train_feature = tf.cast(input_x_train, tf.float32)
            input_x_test = test_set.dynamic_features
            input_x_test = tf.cast(input_x_test, tf.float32)
            for disc in range(n_disc):
                context_state = encode_context(input_x_train_feature)
                fake_input = generator(context_state)
                d_real_pre, d_fake_pre = discriminator(input_x_train_feature, fake_input)

                d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
                d_real_pre_ = tf.reshape(d_real_pre, [-1, 1])

                d_real_pre_loss = cross_entropy(tf.ones_like(d_real_pre_), d_real_pre_)
                d_fake_pre_loss = cross_entropy(tf.zeros_like(d_fake_pre_), d_fake_pre_)

                d_loss = d_real_pre_loss + d_fake_pre_loss
                for weight in discriminator.trainable_variables:
                    d_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                gradient_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
                print('--------------dis_loss-------------', d_loss)
                # print(list(i.name for i in discriminator.trainable_variables))

            d_real_pre, d_fake_pre = discriminator(input_x_train_feature, fake_input)
            d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
            feature_mse = tf.reduce_mean(tf.keras.losses.mae(tf.reshape(input_x_train_feature[:, 3:, :],
                                                                        [-1, feature_dims]),
                                                             tf.reshape(fake_input, [-1, feature_dims])), axis=0)
            mae_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], fake_input))
            gen_loss = lambda_balance * mae_loss + cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_)
            for weight in generator.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            gradient_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # print(list(i.name for i in generator.trainable_variables))
            generator_optimizer.apply_gradients(zip(gradient_generator, generator.trainable_variables))

            print('----------gan_loss--------------', gen_loss)
            print("-----------mse_loss------------", mae_loss)

        context_state_test = encode_context(input_x_test)
        fake_input_test = generator(context_state_test)
        d_real_pre_test, d_fake_pre_test = discriminator(input_x_test, fake_input_test)
        d_real_pre_test_loss = cross_entropy(tf.ones_like(d_real_pre_test), d_real_pre_test)
        d_fake_pre_test_loss = cross_entropy(tf.zeros_like(d_fake_pre_test), d_fake_pre_test)
        d_loss_ = d_real_pre_test_loss + d_fake_pre_test_loss
        gen_loss_ = cross_entropy(tf.ones_like(d_fake_pre_test), d_fake_pre_test)
        input_r = input_x_test[:, 3:, :]
        input_f = fake_input_test
        np.savetxt('real_trajectory_{}.csv'.format(i), input_r.numpy().reshape(-1, feature_dims), delimiter=',')
        np.savetxt('fake_trajectory_{}.csv'.format(i), input_f.numpy().reshape(-1, feature_dims), delimiter=',')
        mse_loss_ = tf.keras.losses.MAE(input_r, input_f)
        feature_mse = tf.reduce_mean(tf.keras.losses.mae(tf.reshape(input_r, [-1, feature_dims]),
                                                         tf.reshape(input_f, [-1, feature_dims])), axis=0)
        # np.savetxt('测试时的feature_mae_{}.csv'.format(i), feature_mse)
        mse_loss_ = mse_loss_
        mse_loss_ = tf.reduce_mean(mse_loss_)
        print("d_loss:{}------gen_loss:{}-------mse_loss:{}---------".format(d_loss_, gen_loss_, mse_loss_))
        return mse_loss_


if __name__ == '__main__':
    data_set = read_gaucoma_data().dynamic_features
    data_set = data_set.reshape(data_set.shape[0], -1)
    data_set_y = np.zeros_like(data_set)
    mse_all = []
    for i in range(5):
        train_x, test_x, train_y, test_y = train_test_split(data_set, data_set_y)
        train_set_ = train_x.reshape(-1, 6, 60)[:, :, 1:]
        test_set_ = test_x.reshape(-1, 6, 60)[:, :, 1:]
        train_set__ = DataSet(train_set_)
        test_set__ = DataSet(test_set_)

        train_set_t = train_x.reshape(-1, 6, 60)[:, :, 0]
        test_set_t = test_x.reshape(-1, 6, 60)[:, :, 0]

        feature_dims = train_set_.shape[2]
        mse_loss = train_step(batch_size=32,
                              hidden_size=128,
                              time_step=6,
                              feature_dims=feature_dims,
                              n_disc=10,
                              train_set=train_set__,
                              test_set=test_set__,
                              epochs=100,
                              lambda_balance=0.005,
                              learning_rate=0.001,
                              l2_regularization=0.001)

        mse_all.append(mse_loss)
    mse_all = np.array(mse_all)
    print(np.mean(mse_all))






