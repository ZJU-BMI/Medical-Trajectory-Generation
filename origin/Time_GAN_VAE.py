import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from modify.data import DataSet
from TimeLSTMCell_1 import *

tf.keras.backend.floatx()
from scipy import stats


# define the discriminator class
class Discriminator(Model):
    def __init__(self, time_step, batch_size, hidden_size, previous_visit, predicted_visit):
        super().__init__(name='discriminator')
        self.time_step = time_step
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit
        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, real_input, fake_input):
        batch = tf.shape(real_input)[0]

        input_same = real_input[:, :self.previous_visit, :]
        input_same_real = input_same
        input_same_fake = input_same

        trajectory_real = []
        trajectory_fake = []

        trajectory_real_predict = tf.zeros(shape=[batch, 0, 1])
        trajectory_fake_predict = tf.zeros(shape=[batch, 0, 1])
        for index in range(self.predicted_visit):
            next_real = real_input[:, index+self.previous_visit, :]
            next_fake = fake_input[:, index, :]
            next_real = tf.reshape(next_real, [batch, 1, -1])
            next_fake = tf.reshape(next_fake, [batch, 1, -1])
            trajectory_step_real = tf.concat((input_same_real, next_real), axis=1)
            trajectory_step_fake = tf.concat((input_same_fake, next_fake), axis=1)

            trajectory_real.append(trajectory_step_real)
            trajectory_fake.append(trajectory_step_fake)

            input_same_real = trajectory_step_real
            input_same_fake = trajectory_step_fake

        for time_index in range(self.predicted_visit):
            trajectory_real_ = trajectory_real[time_index]
            trajectory_fake_ = trajectory_fake[time_index]

            state = self.LSTM_Cell.get_initial_state(batch_size=batch, dtype=tf.float32)
            state_real = state
            state_fake = state
            for t in range(tf.shape(trajectory_real_)[1]):
                input_real = trajectory_real_[:, t, :]
                input_fake = trajectory_fake_[:, t, :]
                output_real, state_real = self.LSTM_Cell(input_real, state_real)
                output_fake, state_fake = self.LSTM_Cell(input_fake, state_fake)

            output_fake = self.dense1(output_fake)
            output_real = self.dense1(output_real)

            trajectory_step_real_pre = self.dense2(output_real)
            trajectory_step_fake_pre = self.dense2(output_fake)

            trajectory_step_real_pre = tf.reshape(trajectory_step_real_pre, [batch, -1, 1])
            trajectory_step_fake_pre = tf.reshape(trajectory_step_fake_pre, [batch, -1, 1])

            trajectory_real_predict = tf.concat((trajectory_real_predict, trajectory_step_real_pre), axis=1)
            trajectory_fake_predict = tf.concat((trajectory_fake_predict, trajectory_step_fake_pre), axis=1)

        return trajectory_real_predict, trajectory_fake_predict


# Define the generator class: get the output y according to last hidden state and attention mechanism
class Generator(Model):
    def __init__(self, z_dims, feature_dims, hidden_size, time_step, previous_visit, predicted_visit):
        super().__init__(name='generator')
        self.feature_dims = feature_dims
        self.hidden_size = hidden_size
        self.z_dims = z_dims
        self.time_step = time_step
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit
        self.LSTM_Cell_decode = TimeLSTMCell_1(hidden_size)
        # parameters for output y
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

        # parameters for z_j, mean_j and log_var_j in posterior network
        self.dense4 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)

        # parameters for z_j, mean_j and log_var_j in prior network
        self.dense7 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense8 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)
        self.dense9 = tf.keras.layers.Dense(units=z_dims, activation=tf.nn.relu)

    def build(self, context_state_shape):
        output_shape = tf.TensorShape((self.hidden_size, self.feature_dims))
        self.output_weight = self.add_weight(name='s_y',
                                             shape=output_shape,
                                             initializer='uniform',
                                             trainable=True)
        super(Generator, self).build(context_state_shape)

    def call(self, context_state):  # 这里考虑了时间因素
        h_i, input_t = context_state
        batch = tf.shape(h_i)[0]
        z_j = tf.zeros(shape=(batch,self.z_dims))
        y_j_ = tf.zeros(shape=(batch, self.feature_dims))
        h_i = tf.reshape(h_i, (-1, self.hidden_size))
        state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=float)
        fake_input = tf.zeros(shape=[batch, 0, self.feature_dims])
        mean_all = tf.zeros(shape=(batch, 0, self.z_dims))
        log_var_all = tf.zeros(shape=[batch, 0, self.z_dims])
        z_all = []
        for time in range(self.predicted_visit):
            input_j = tf.concat((z_j, h_i, y_j_), axis=1)
            input_j = tf.reshape(input_j, [-1, self.z_dims+self.hidden_size+self.feature_dims])
            t = input_t[:, time+self.previous_visit, :]
            t = tf.reshape(t, [-1, 1])
            output, state = self.LSTM_Cell_decode([input_j, t], state)

            y_j = tf.reshape(output, [batch, -1])
            y_j = self.dense1(y_j)
            y_j = self.dense2(y_j)
            y_j = self.dense3(y_j)
            z_j, mean_j, log_var_j = self.inference_network(batch=batch, h_i=h_i, s_j=state[1], y_j=y_j, y_j_1=y_j_)

            fake_input = tf.concat((fake_input, tf.reshape(y_j, [-1, 1, self.feature_dims])), axis=1)
            mean_all = tf.concat((mean_all, tf.reshape(mean_j, [-1, 1, self.z_dims])), axis=1)
            log_var_all = tf.concat((log_var_all, tf.reshape(log_var_j, [-1, 1, self.z_dims])), axis=1)

            z_j_, mean_j_, log_var_j_ = self.prior_network(batch=batch, h_i=h_i, s_j=state[1], y_j_1=y_j_)
            z_all.append([z_j, z_j_])

            y_j_ = y_j

        return tf.reshape(fake_input, [-1, self.predicted_visit, self.feature_dims]), mean_all, log_var_all, z_all

    # inference network
    def inference_network(self, batch, h_i, s_j, y_j, y_j_1):
        inference_input = tf.concat((h_i, s_j, y_j, y_j_1), axis=1)  # [batch_size, 2*hidden_size+2*feature_dims]
        h_z_j = self.dense4(inference_input)
        mean_j = self.dense5(h_z_j)
        log_var_j = self.dense6(h_z_j)
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(self.z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z_j = mean_j + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var_j)))  # [batch, z_dims]
        return z_j, mean_j, log_var_j

    # prior network
    def prior_network(self, batch, h_i, s_j, y_j_1):
        prior_input = tf.concat((h_i, s_j, y_j_1), axis=1)
        h_z_j_ = self.dense7(prior_input)
        mean_j_ = self.dense8(h_z_j_)
        log_var_j_ = self.dense9(h_z_j_)
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(self.z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z_j_ = mean_j_ + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var_j_)))  # [batch, z_dims]
        return z_j_, mean_j_, log_var_j_


class EncodeContext(Model):
    def __init__(self, hidden_size, time_step, batch_size, previous_visit):
        super().__init__(name='encode_context')
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.batch_size = batch_size
        self.previous_visit = previous_visit
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_context):
        batch = tf.shape(input_context)[0]
        state = self.LSTM_Cell_encode.get_initial_state(batch_size=batch, dtype=tf.float32)
        context_state = np.zeros(shape=[batch, 0, self.hidden_size])
        for time in range(self.previous_visit):
            input = input_context[:, time, :]
            output, state = self.LSTM_Cell_encode(input, state)
            output = tf.reshape(output, [batch, 1, -1])
            context_state = np.concatenate((context_state, output), axis=1)
        return context_state


def train_step(hidden_size, n_disc, lambda_balance, learning_rate, l2_regularization, imbalance_kl, z_dims, t_imbalance):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)
    # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)

    train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)
    # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)
    test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)

    # train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)
    # test_set = np.load('HF_validate_.npy').reshape(-1, 6, 30)

    time_step = 6
    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0
    previous_visit = 3
    predicted_visit = 1

    batch_size = 32
    epochs = 1

    # hidden_size = 2**(int(hidden_size))
    # z_dims = 2 ** (int(z_dims))
    # n_disc = int(n_disc)
    # lambda_balance = 10**lambda_balance
    # learning_rate = 10**learning_rate
    # l2_regularization = 10**l2_regularization
    # imbalance_kl = 10 ** imbalance_kl
    # t_imbalance = 10 ** t_imbalance

    print('previous_visit---{}----predicted—_visit{}'.format(previous_visit, predicted_visit))
    print('----batch_size{}---hidden_size{}---n_disc{}---epochs{}---'
          'lambda_balance{}---learning_rate{}---l2_regularization{}---kl_imbalance{}---z_dims{}---'
          .format(batch_size, hidden_size, n_disc, epochs, lambda_balance, learning_rate, l2_regularization,
                  imbalance_kl, z_dims))

    discriminator = Discriminator(time_step=time_step, batch_size=batch_size, hidden_size=hidden_size, previous_visit=previous_visit, predicted_visit=predicted_visit)
    generator = Generator(feature_dims=feature_dims,
                          hidden_size=hidden_size,
                          z_dims=z_dims,
                          time_step=time_step,
                          previous_visit=previous_visit,
                          predicted_visit=predicted_visit)

    encode_context = EncodeContext(hidden_size=hidden_size,
                                   time_step=time_step,
                                   batch_size=batch_size,
                                   previous_visit=previous_visit)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        l = 0
        while train_set.epoch_completed < epochs:
            l += 1
            input_x_train = train_set.next_batch(batch_size)
            input_x_train_feature = tf.cast(input_x_train[:, :, 1:], tf.float32)
            input_t_train = input_x_train[:, :, 0].reshape(-1, 6, 1)
            input_t_train = tf.cast(input_t_train, tf.float32)

            for disc in range(n_disc):
                context_state = encode_context(input_x_train_feature)
                h_i = tf.reshape(context_state[:, -1, :], [-1, hidden_size])
                fake_input, mean_all, log_var_all, z_all = generator([h_i, input_t_train])
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

            print("第{}批次的数据训练完毕".format(l))
            d_real_pre, d_fake_pre = discriminator(input_x_train_feature, fake_input)
            d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])

            mae_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, previous_visit:previous_visit+predicted_visit, :], fake_input))

            kl_loss_all = []
            KL = tf.keras.losses.KLDivergence()
            for m in range(len(z_all)):
                posterior_d, prior_d = z_all[m]
                kl_loss = -KL(posterior_d, prior_d)
                kl_loss_all.append(kl_loss)
            kl_loss_all = tf.reduce_mean(kl_loss_all)

            gen_loss = mae_loss + cross_entropy(tf.ones_like(d_fake_pre_), d_fake_pre_) * lambda_balance +\
                       imbalance_kl * kl_loss_all

            variables = [var for var in generator.trainable_variables]
            for var in encode_context.trainable_variables:
                variables.append(var)

            for weight in generator.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in encode_context.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient_generator = gen_tape.gradient(gen_loss, variables)
            generator_optimizer.apply_gradients(zip(gradient_generator, variables))

        print("开始测试！")
        input_x_test_all = tf.zeros(shape=(0, 6, feature_dims))
        generated_x_test_all = tf.zeros(shape=(0, predicted_visit, feature_dims))
        while test_set.epoch_completed < epochs:
            input_test = test_set.next_batch(batch_size)
            input_test_x = input_test[:, :, 1:]
            input_test_x = tf.cast(input_test_x, tf.float32)
            input_test_t = np.reshape(input_test[:, :, 0], [-1, time_step, 1])
            context_state_test = encode_context(input_test_x)
            h_i_test = tf.reshape(context_state_test[:, -1, :], [-1, hidden_size])
            fake_input_test, mean_all_test, log_var_all_test, z_all_test = generator([h_i_test, input_test_t])
            input_x_test_all = tf.concat((input_x_test_all, input_test_x), axis=0)
            generated_x_test_all = tf.concat((generated_x_test_all, fake_input_test), axis=0)

        mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_test_all[:, previous_visit:previous_visit+predicted_visit, :], generated_x_test_all))
        r_value_all = []
        p_value_all = []
        for r in range(predicted_visit):
            x_ = tf.reshape(input_x_test_all[:, previous_visit + r, :], (-1,))
            y_ = tf.reshape(generated_x_test_all[:, r, :], (-1,))
            r_value_ = stats.pearsonr(x_, y_)
            r_value_all.append(r_value_[0])
            p_value_all.append(r_value_[1])
        print('-----------r_value{}----------'.format(np.mean(r_value_all)))
        print('------------p_value{}-----------'.format(np.mean(p_value_all)))
        print("------mse_loss:{}---------".format(mse_loss))
        tf.compat.v1.reset_default_graph()
        return -1*mse_loss.numpy(), np.mean(r_value_all)
        # return -1*mse_loss


if __name__ == '__main__':
    # GAN_time_LSTM_BO = BayesianOptimization(
    #     train_step, {
    #         'hidden_size': (3, 4),
    #         'z_dims': (3, 4),
    #         'n_disc': (1, 10),
    #         'lambda_balance': (-6, 0),
    #         'imbalance_kl': (-6, 0),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #         't_imbalance': (-6, -1)
    #     }
    # )
    # GAN_time_LSTM_BO.maximize()
    # print(GAN_time_LSTM_BO.max)

    # # 青光眼数据
    # mse_all = []
    # for i in range(10):
    #     mse = train_step(hidden_size=16, n_disc=8, lambda_balance=0.00001133371266955815, learning_rate=0.028435345041478252, l2_regularization=0.00043666445620540984,
    #                      imbalance_kl=0.000004893786395464466, z_dims=16)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    # MIMIC 数据
    mse_all = []
    r_value_all = []
    for i in range(50):
        mse, r_value = train_step(hidden_size=8, n_disc=7, lambda_balance=0.007510547275533875, learning_rate=0.016357346440746212, l2_regularization=0.000527042245026845,imbalance_kl=4.278988766353393e-05,z_dims=8,t_imbalance=0.0)
        mse_all.append(mse)
        r_value_all.append(r_value)
    print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    print('----------------mse_std:{}----------'.format(np.std(mse_all)))
    print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
    print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

    # HF 数据
    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=16, n_disc=10, lambda_balance=0.0000012853797171914746, learning_rate=0.03169221116390897, l2_regularization=0.000013305878437453292, z_dims=16, imbalance_kl=0.00002208275629978935)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))