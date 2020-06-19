import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet, read_data, read_gaucoma_data
from LSTMCell import *
from bayes_opt import BayesianOptimization

# define the discriminator class
class Discriminator(Model):
    def __init__(self, time_step, batch_size):
        super().__init__(name='discriminator')
        self.time_step = time_step
        self.batch_size = batch_size
        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        self.dense3 = tf.keras.layers.Dense(units=20)
        self.dense4 = tf.keras.layers.Dense(units=20)

    def call(self, real_input, fake_input):
        batch = tf.shape(real_input)[0]
        input_same = real_input[:, :self.time_step - 3, :]
        trajectory_real = []
        trajectory_fake = []
        trajectory_real_predict = tf.zeros(shape=[batch, 0])
        trajectory_fake_predict = tf.zeros(shape=[batch, 0])
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
            trajectory_step_real = self.dense3(trajectory_step_real)
            trajectory_step_real = self.dense4(trajectory_step_real)
            trajectory_step_real_predict = self.dense2(trajectory_step_real)

            trajectory_step_fake = self.dense1(trajectory_step_fake)
            trajectory_step_fake = self.dense3(trajectory_step_fake)
            trajectory_step_fake = self.dense4(trajectory_step_fake)
            trajectory_step_fake_predict = self.dense2(trajectory_step_fake)

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
        self.dense1 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

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
        last_hidden = tf.reshape(last_hidden, (-1, self.hidden_size))
        action_var_all = tf.zeros(shape=(batch, 0, self.feature_dims))
        norm_dist_all = []
        for time in range(self.time_step-3):
            action_var_step = tf.zeros(shape=(batch, 0))
            norm_dist_step = []
            if time == 0:
                state = last_hidden
            for dim in range(self.feature_dims):
                mu = self.dense1(state)
                sigma = self.dense2(state)
                sigma = tf.nn.softplus(sigma) + 1e-8
                mu = tf.squeeze(mu)
                sigma = tf.squeeze(sigma)
                norm_dist = tf.compat.v1.distributions.Normal(mu, sigma, allow_nan_stats=False)
                action_var = norm_dist.sample(1)
                action_var = tf.clip_by_value(action_var, 0, 1)
                action_var_step = tf.concat((action_var_step, tf.reshape(action_var, (batch, -1))), axis=1)
                norm_dist_step.append(norm_dist)
            norm_dist_all.append(norm_dist_step)
            action_var_step_ = tf.reshape(action_var_step, (-1, 1, self.feature_dims))
            if time == 0:
                c_h = self.LSTM_Cell_encode.get_initial_state(batch_size=batch, dtype=tf.float32)
            else:
                c_h = new_c_h
            state, new_c_h = self.LSTM_Cell_encode(tf.reshape(action_var_step_, [-1, self.feature_dims]), c_h)
            action_var_all = tf.concat((action_var_all, action_var_step_), axis=1)
        return norm_dist_all, action_var_all


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


def reward_discount(context_state, discount):
    batch = tf.shape(context_state)[0]
    discount_matrix = discount * tf.ones(shape=(batch, 1))
    recurrent_matrix = discount * tf.ones(shape=(batch, 1))
    for i in range(2):
        new_column = discount*recurrent_matrix
        discount_matrix = tf.concat((discount_matrix, new_column), axis=1)
    return discount_matrix


def train_step(hidden_size, n_disc, lambda_balance_cross_entropy,
               learning_rate, l2_regularization):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)[:, :, 1:]
    # # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)[:, :, 1:]

    # train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)[:, :, 1:]
    # # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)[:, :, 1:]
    # test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)[:, :, 1:]

    # train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)[:, :, 1:]
    # test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)[:, :, 1:]
    # # test_set = np.load('HF_validate_.npy').reshape(-1, 6, 30)[:, :, 1:]

    train_set = np.load('generate_train_x_.npy').reshape(-1, 6, 30)[:, :, 0:1]
    test_set = np.load('generate_validate_x_.npy').reshape(-1, 6, 30)[:, :, 0:1]
    # test_set = np.load('generate_test_x_.npy').reshape(-1, 6, 30)
    time_step = 6
    feature_dims = train_set.shape[2]

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0

    batch_size = 64
    epochs = 100

    hidden_size = 2**(int(hidden_size))
    n_disc = int(n_disc)
    lambda_balance_cross_entropy = 10 ** lambda_balance_cross_entropy
    learning_rate = 10**learning_rate
    l2_regularization = 10**l2_regularization

    discount = 0.99
    lambda_balance_gen = 1

    print('----batch_size{}---hidden_size{}---n_disc{}---epochs{}---'
          'lambda_balance_gen{}---lambda_imbalance_cross_entropy{}-----'
          'learning_rate{}---l2_regularization{}---discount{}----'
          .format(batch_size, hidden_size, n_disc, epochs, lambda_balance_gen, lambda_balance_cross_entropy,
                  learning_rate, l2_regularization, discount))

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

    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        while train_set.epoch_completed < epochs:
            input_x_train = train_set.next_batch(batch_size)
            input_x_train_feature = tf.cast(input_x_train, tf.float32)
            input_x_test = test_set.dynamic_features
            input_x_test = tf.cast(input_x_test, tf.float32)
            for disc in range(n_disc):
                context_state = encode_context(input_x_train_feature)
                norm_dist, action_all = generator(context_state)
                d_real_pre, d_fake_pre = discriminator(input_x_train_feature, action_all)

                d_fake_pre_ = tf.reshape(d_fake_pre, [-1, 1])
                d_real_pre_ = tf.reshape(d_real_pre, [-1, 1])

                d_real_pre_loss = cross_entropy(tf.ones_like(d_real_pre_), d_real_pre_)
                d_fake_pre_loss = cross_entropy(tf.zeros_like(d_fake_pre_), d_fake_pre_)

                d_loss = d_real_pre_loss + d_fake_pre_loss
                for weight in discriminator.trainable_variables:
                    d_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                gradient_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
                # print('***********第{}次训练鉴别器结束**************'.format(disc))

            discount_matrix = reward_discount(input_x_train_feature, discount)
            prob_all = tf.ones(shape=(tf.shape(discount_matrix)[0], 0))
            entropy_all = tf.zeros(shape=(tf.shape(discount_matrix)[0], 0))
            for time_index in range(len(norm_dist)):
                entropy = 0
                prob_time = tf.ones(shape=(tf.shape(discount_matrix)[0], 1))
                time_norm_dist = norm_dist[time_index]
                for dim_index in range(len(time_norm_dist)):
                    dist = time_norm_dist[dim_index]
                    entropy += dist.entropy()
                    action = action_all[:, time_index, dim_index]
                    prob = tf.math.log(dist.prob(action) + 1e-8)
                    prob_time = prob_time * tf.reshape(prob, (-1, 1))
                entropy_all = tf.concat((entropy_all, tf.reshape(entropy, (-1, 1))), axis=1)
                prob_all = tf.concat((prob_all, prob_time), axis=1)

            mse = tf.keras.losses.mse(input_x_train_feature[:, 3:, :], action_all)
            # print(mse)
            gen_loss = -tf.reduce_mean(mse*discount_matrix*tf.math.log(d_fake_pre))
            cross_entropy_loss = -tf.reduce_mean(tf.math.exp(prob_all)*prob_all)
            # entropy_all_loss = tf.reduce_mean(entropy_all)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, 3:, :], action_all))
            # gen_loss = gen_loss_ + cross_entropy_loss*lambda_balance_cross_entropy

            # print("------gen_loss:{}-------mse_loss:{}----cross_entropy{}-----".format(gen_loss_, mse_loss, cross_entropy_loss))

            for weight in generator.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in encode_context.trainable_variables:
                gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = [var for var in generator.trainable_variables]
            for var in encode_context.trainable_variables:
                variables.append(var)
            gradient_generator = gen_tape.gradient(gen_loss, variables)
            generator_optimizer.apply_gradients(zip(gradient_generator, variables))

        print('##############开始测试###################')
        context_state = encode_context(input_x_test)
        norm_dist_test, action_all_test = generator(context_state)
        d_real_pre_test, d_fake_pre_test = discriminator(input_x_test, action_all_test)
        d_real_pre_test_loss = cross_entropy(tf.ones_like(d_real_pre_test), d_real_pre_test)
        d_fake_pre_test_loss = cross_entropy(tf.zeros_like(d_fake_pre_test), d_fake_pre_test)
        d_loss_ = d_real_pre_test_loss + d_fake_pre_test_loss
        discount_matrix = reward_discount(input_x_test, discount)
        prob_all_test = tf.ones(shape=(tf.shape(discount_matrix)[0], 0))
        entropy_all = tf.zeros(shape=(tf.shape(discount_matrix)[0], 0))
        for time_index in range(len(norm_dist_test)):
            entropy = 0
            prob_time = tf.ones(shape=(tf.shape(discount_matrix)[0], 1))
            time_norm_dist = norm_dist_test[time_index]
            for dim_index in range(len(time_norm_dist)):
                dist = time_norm_dist[dim_index]
                entropy += dist.entropy()
                action = action_all_test[:, time_index, dim_index]
                prob = tf.math.log(dist.prob(action) + 1e-8)
                prob_time = prob_time * tf.reshape(prob, (-1, 1))
            prob_all_test = tf.concat((prob_all_test, prob_time), axis=1)
            entropy_all_test = tf.concat((entropy_all, tf.reshape(entropy, (-1, 1))), axis=1)

        mse = tf.keras.losses.mse(input_x_test[:, 3:, :], action_all_test)
        gen_loss = -tf.reduce_mean(mse * discount_matrix * tf.math.log(d_fake_pre_test))

        cross_entropy_loss = -tf.reduce_mean(tf.math.exp(prob_all_test) * prob_all_test)

        mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, 3:, :], action_all_test))

        print("d_loss:{}------gen_loss:{}-------mse_loss:{}---cross_entropy{}-----"
              "-".format(d_loss_, gen_loss, mse_loss, cross_entropy_loss))
        tf.compat.v1.reset_default_graph()
        return -1*mse_loss


if __name__ == '__main__':
    GAN_LSTM_BO = BayesianOptimization(
        train_step, {
            'hidden_size': (5, 7),
            'n_disc': (1, 10),
            'lambda_balance_cross_entropy': (-6, 0),
            'learning_rate': (-5, -1),
            'l2_regularization': (-5, -1)
        }
    )
    GAN_LSTM_BO.maximize()
    print(GAN_LSTM_BO.max)
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
    #     mse = train_step(hidden_size=64, n_disc=9, lambda_balance_cross_entropy=0.0013124872591961127,
    #                      learning_rate=0.05717309442222128, l2_regularization=0.048080745538964575)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))



