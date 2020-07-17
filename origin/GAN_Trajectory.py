import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from modify.data import DataSet
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


def train_step(hidden_size, n_disc, lambda_balance, learning_rate, l2_regularization):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)[:, :, 1:]
    # # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)[:, :, 1:]

    # train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)[:, :, 1:]
    # # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)[:, :, 1:]
    # test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)[:, :, 1:]

    train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)[:, :, 1:]
    test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)[:, :, 1:]
    # test_set = np.load('HF_validate_.npy').reshape(-1, 6, 30)[:, :, 1:]

    train_set = np.load('generate_train_x_.npy').reshape(-1, 6, 30)
    # test_set = np.load('generate_validate_x_.npy').reshape(-1, 6, 30)
    test_set = np.load('generate_test_x_.npy').reshape(-1, 6, 30)

    time_step = 6
    feature_dims = train_set.shape[2]

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0

    batch_size = 64
    epochs = 1

    # hidden_size = 2**(int(hidden_size))
    # n_disc = int(n_disc)
    # lambda_balance = 10**lambda_balance
    # learning_rate = 10**learning_rate
    # l2_regularization = 10**l2_regularization

    print('----batch_size{}---hidden_size{}---n_disc{}---epochs{}---'
          'lambda_balance{}---learning_rate{}---l2_regularization{}---'
          .format(batch_size, hidden_size, n_disc, epochs, lambda_balance, learning_rate, l2_regularization))

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
        while train_set.epoch_completed < epochs:
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

            d_real_pre, d_fake_pre = discriminator(input_x_train_feature, fake_input)
            d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
            mae_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], fake_input))
            mse = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], fake_input), axis=0)
            print(mse)
            gen_loss = mae_loss + cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_)*lambda_balance

            for weight in generator.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in encode_context.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = [var for var in generator.trainable_variables]
            for var in encode_context.trainable_variables:
                variables.append(var)

            gradient_generator = gen_tape.gradient(gen_loss, variables)
            generator_optimizer.apply_gradients(zip(gradient_generator, variables))

        context_state_test = encode_context(input_x_test)
        fake_input_test = generator(context_state_test)
        d_real_pre_test, d_fake_pre_test = discriminator(input_x_test, fake_input_test)
        d_real_pre_test_loss = cross_entropy(tf.ones_like(d_real_pre_test), d_real_pre_test)
        d_fake_pre_test_loss = cross_entropy(tf.zeros_like(d_fake_pre_test), d_fake_pre_test)
        d_loss_ = d_real_pre_test_loss + d_fake_pre_test_loss
        gen_loss_ = cross_entropy(tf.ones_like(d_fake_pre_test), d_fake_pre_test)
        input_r = input_x_test[:, 3:, :]
        input_f = fake_input_test
        mse_loss_ = tf.reduce_mean(tf.keras.losses.mse(input_r, input_f))
        mse = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], fake_input), axis=0)
        print(mse)
        print("d_loss:{}------gen_loss:{}-------mse_loss:{}---------".format(d_loss_, gen_loss_, mse_loss_))
        tf.compat.v1.reset_default_graph()
        return -1*mse_loss_


if __name__ == '__main__':
    # GAN_LSTM_BO = BayesianOptimization(
    #     train_step, {
    #         'hidden_size': (5, 7),
    #         'n_disc': (10, 20),
    #         'lambda_balance': (-6, 0),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #     }
    # )
    # GAN_LSTM_BO.maximize()
    # print(GAN_LSTM_BO.max)
    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=64, n_disc=15, lambda_balance=0.808, learning_rate=0.02887, l2_regularization=0.0012)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))

    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=64, n_disc=12, lambda_balance=0.001607, learning_rate=0.02547, l2_regularization=0.0010521)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=32, n_disc=15, lambda_balance=0.000001, learning_rate=0.1, l2_regularization=0.0004803126481777589)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    mse_all = []
    for i in range(50):
        mse = train_step(hidden_size=32, n_disc=19, lambda_balance=0.5800768549089896, learning_rate=0.09967514396270517, l2_regularization=0.016098856440393123)
        mse_all.append(mse)
        print('第{}次测试完成'.format(i))
    print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    print('----------------mse_std:{}----------'.format(np.std(mse_all)))

