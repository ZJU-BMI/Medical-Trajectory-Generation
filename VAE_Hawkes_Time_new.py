import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from data import DataSet
from LSTMCell import *
import sys
import os
from bayes_opt import BayesianOptimization
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Define encode class: encode the history information to deep representation[batch_size, 3, hidden_size]
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


# input: z_j:[batch,z_dim], h_i:[batch,hidden_size],y_j:[batch,feature_dims],y_(j_1):[batch,feature_dims]
# output: generated_visit:[batch,k,feature_dims]
class Decoder(Model):
    def __init__(self, feature_dims, z_dims, hidden_size, time_step, previous_visit, predicted_visit):
        super(Decoder, self).__init__(name='Decoder_network')
        self.feature_dims = feature_dims
        self.z_dims = z_dims
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit
        self.LSTM_Cell_decode = LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=self.feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=self.feature_dims, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)

        self.dense7 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)
        self.dense8 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)
        self.dense9 = tf.keras.layers.Dense(units=self.z_dims, activation=tf.nn.relu)

    def build(self, context_state_shape):
        beta_init = tf.random_normal_initializer(mean=0.8, stddev=0.05, seed=42)
        base_intensity_init = tf.random_normal_initializer(mean=0.2, stddev=0.1, seed=2)
        output_shape = tf.TensorShape((self.hidden_size, self.feature_dims))
        shape_weight_hawkes = tf.TensorShape((1, 1))
        self.output_weight = self.add_weight(name='s_y',
                                             shape=output_shape,
                                             initializer='uniform',
                                             trainable=True)

        self.trigger_parameter_beta = tf.Variable(initial_value=beta_init(shape=shape_weight_hawkes),
                                                  name='trigger_parameter_beta')

        self.trigger_parameter_alpha = self.add_weight(name='trigger_parameter_alpha',
                                                       initializer='uniform',
                                                       shape=shape_weight_hawkes,
                                                       trainable=True)

        self.base_intensity = tf.Variable(initial_value=base_intensity_init(shape=shape_weight_hawkes),
                                          name='base_intensity_parameter')

        super(Decoder, self).build(context_state_shape)

    def call(self, context_state):
        h_i, input_t = context_state
        batch = tf.shape(h_i)[0]
        z_j = tf.zeros(shape=(batch, self.z_dims))
        y_j_ = tf.zeros(shape=(batch, self.feature_dims))
        h_i = tf.reshape(h_i, (-1, self.hidden_size))
        state = self.LSTM_Cell_decode.get_initial_state(batch_size=batch, dtype=float)
        fake_input = tf.zeros(shape=[batch, 0, self.feature_dims])
        mean_all_post = tf.zeros(shape=[batch, 0, self.z_dims])
        log_var_all_post = tf.zeros(shape=[batch, 0, self.z_dims])
        mean_all_prior = tf.zeros(shape=[batch, 0, self.z_dims])
        log_var_all_prior = tf.zeros(shape=[batch, 0, self.z_dims])
        z_all = []
        time_estimate_probability_all = tf.zeros(shape=(0, 1))
        for j in range(self.predicted_visit):
            input_j = tf.concat((z_j, h_i, y_j_), axis=1)
            input_j = tf.reshape(input_j, [batch, self.hidden_size+self.feature_dims+self.z_dims])
            current_time_index = j+self.previous_visit
            output, state = self.LSTM_Cell_decode(input_j, state)
            c = state[0]
            h_ = state[1]
            trigger_parameter_beta = tf.tile(self.trigger_parameter_beta, [batch, 1])
            trigger_parameter_alpha = tf.tile(self.trigger_parameter_alpha, [batch, 1])
            base_intensity = tf.tile(self.base_intensity, [batch, 1])
            condition_intensity_value = self.calculate_hawkes_process(batch=batch, input_t=input_t,
                                                                      current_time_index=current_time_index,
                                                                      trigger_parameter_alpha=trigger_parameter_alpha,
                                                                      trigger_parameter_beta=trigger_parameter_beta,
                                                                      base_intensity=base_intensity)

            probability_time = self.calculate_time_intervals_probability(batch=batch, input_t=input_t,
                                                                         current_time_index=current_time_index,
                                                                         trigger_parameter_alpha=trigger_parameter_alpha,
                                                                         trigger_parameter_beta=trigger_parameter_beta,
                                                                         base_intensity=base_intensity)
            time_estimate_probability_all = tf.concat((time_estimate_probability_all, probability_time), axis=0)

            condition_intensity_value_ = tf.tile(tf.reshape(condition_intensity_value, [batch, 1]),
                                                 [1, self.hidden_size])

            h = condition_intensity_value_ * h_
            state = [c, h]
            y_j = tf.reshape(output, [batch, -1])
            y_j = self.dense1(y_j)
            y_j = self.dense2(y_j)
            y_j = self.dense3(y_j)
            z_j, mean_j, log_var_j = self.inference_network(batch=batch, h_i=h_i, s_j=h_, y_j=y_j, y_j_1=y_j_)

            fake_input = tf.concat((fake_input, tf.reshape(y_j, [-1, 1, self.feature_dims])), axis=1)
            mean_all_post = tf.concat((mean_all_post, tf.reshape(mean_j, [-1, 1, self.z_dims])), axis=1)
            log_var_all_post = tf.concat((log_var_all_post, tf.reshape(log_var_j, [-1, 1, self.z_dims])), axis=1)
            z_j_, mean_j_, log_var_j_ = self.prior_network(batch=batch, h_i=h_i, s_j=h_, y_j_1=y_j_)

            mean_all_prior = tf.concat((mean_all_prior, tf.reshape(mean_j_,  [-1, 1, self.z_dims])), axis=1)
            log_var_all_prior = tf.concat((log_var_all_prior, tf.reshape(log_var_j_, [-1, 1, self.z_dims])), axis=1)

            z_all.append([z_j, z_j_])

            y_j_ = y_j
        return tf.reshape(fake_input, [-1, self.predicted_visit, self.feature_dims]), mean_all_post, log_var_all_post, mean_all_prior, log_var_all_prior, z_all, time_estimate_probability_all

    # 计算lambda_t函数表达式：lambda_0 + alpha sum_{t_j < t) exp(-beta(t-t_j))
    def calculate_hawkes_process(self, batch, input_t, current_time_index,
                                 trigger_parameter_beta, trigger_parameter_alpha, base_intensity):

        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])  # number value[batch ,1]
        current_t_tile = tf.tile(current_t, [1, current_time_index])  # [batch, current_index]

        time_before_t = input_t[:, :current_time_index]
        time_difference = time_before_t - current_t_tile  # [batch, current_index]

        triggering_kernel = tf.reduce_sum(tf.exp(time_difference * trigger_parameter_beta), axis=1)

        triggering_kernel = tf.reshape(triggering_kernel, [batch, 1])
        condition_intensity_value = base_intensity + trigger_parameter_alpha * triggering_kernel
        return condition_intensity_value

    # 计算 the probability of event occurring at last+time_intervals: max(f(delta_t))
    def calculate_time_intervals_probability(self, batch, input_t, current_time_index,
                                             trigger_parameter_beta, trigger_parameter_alpha, base_intensity):

        ratio_alpha_beta = trigger_parameter_alpha / trigger_parameter_beta
        # the current time
        current_time = input_t[:, current_time_index]
        current_time = tf.reshape(current_time, [batch, -1])

        time_before_t = input_t[:, :current_time_index]

        current_time_tile = tf.tile(current_time, [1, current_time_index])  # [batch, current_time_index]

        # part_1: t_i - delta_t
        time_difference_1 = time_before_t - current_time_tile
        trigger_kernel = tf.reduce_sum(tf.exp(trigger_parameter_beta * time_difference_1), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity = base_intensity + trigger_parameter_alpha * trigger_kernel  # [batch, 1]

        # part_2: t_{j_1} - delta_t
        last_time = tf.reshape(input_t[:, current_time_index-1], [batch, 1])
        time_difference_2 = (last_time - current_time) * base_intensity  # [batch, 1]

        # part_3: t_i - t_{j-1}
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = tf.reduce_sum(tf.exp(trigger_parameter_beta * (time_before_t - last_time_tile)), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, -1])

        probability_result = condition_intensity * tf.exp(time_difference_2 + ratio_alpha_beta*(trigger_kernel- time_difference_3))

        probability_result = tf.reshape(probability_result, [batch, 1])
        # print('f_t--------{}',format(probability_result))

        return probability_result

    def inference_network(self, batch, h_i, s_j, y_j, y_j_1):
        inference_input = tf.concat((h_i, s_j, y_j, y_j_1), axis=1)  # [batch_size, 2*hidden_size+2*feature_dims]
        # inference_input = tf.reshape(inference_input, [1, -1])
        h_z_j = self.dense4(inference_input)
        mean_j = self.dense5(h_z_j)
        log_var_j = self.dense6(h_z_j)
        mean_j = tf.reshape(mean_j, [-1, self.z_dims])
        log_var_j = tf.reshape(log_var_j, [-1, self.z_dims])
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(self.z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z_j = mean_j + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var_j)))  # [batch, z_dims]
        return z_j, mean_j, log_var_j

    def prior_network(self, batch, h_i, s_j, y_j_1):
        prior_input = tf.concat((h_i, s_j, y_j_1), axis=1)
        # prior_input = tf.reshape(prior_input, [1, -1])
        h_z_j_ = self.dense7(prior_input)
        mean_j_ = self.dense8(h_z_j_)
        log_var_j_ = self.dense9(h_z_j_)
        mean_j_ = tf.reshape(mean_j_, [-1, self.z_dims])
        log_var_j_ = tf.reshape(log_var_j_, [-1, self.z_dims])
        sample_all = tf.zeros(shape=(batch, 0))
        for feature in range(self.z_dims):
            sample = tf.compat.v1.random_normal(shape=(batch, 1))
            sample_all = tf.concat((sample_all, sample), axis=1)
        z_j_ = mean_j_ + tf.multiply(sample_all, tf.math.sqrt(tf.exp(log_var_j_)))  # [batch, z_dims]
        return z_j_, mean_j_, log_var_j_


def train_step(hidden_size, n_disc, learning_rate, l2_regularization, imbalance_kl, z_dims, t_imbalance):
    # train_set = np.load('train_x_.npy').reshape(-1, 6, 60)
    # test_set = np.load('test_x.npy').reshape(-1, 6, 60)
    # test_set = np.load('validate_x_.npy').reshape(-1, 6, 60)

    # train_set = np.load('mimic_train_x_.npy').reshape(-1, 6, 37)
    # test_set = np.load('mimic_validate_.npy').reshape(-1, 6, 37)
    # test_set = np.load('mimic_test_x_.npy').reshape(-1, 6, 37)

    train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)
    # test_set = np.load('HF_validate_.npy').reshape(-1, 6, 30)

    # train_set = np.load('generate_train_x_.npy').reshape(-1, 6, 30)
    # # test_set = np.load('generate_validate_x_.npy').reshape(-1, 6, 30)
    # test_set = np.load('generate_test_x_.npy').reshape(-1, 6, 30)

    time_train = train_set[:, :, 0]
    batch_train = time_train.shape[0]
    time_train_ = time_train.reshape([-1, 1])
    scaler = MinMaxScaler()
    scaler.fit(time_train_)
    feature_normalization = scaler.transform(time_train_)
    feature_normalization = feature_normalization.reshape(batch_train, 6, 1)
    train_set = np.concatenate((train_set[:, :, 1:], feature_normalization), axis=2)

    time_test = test_set[:, :, 0]
    batch_test = time_test.shape[0]
    time_test_ = time_test.reshape([-1, 1])
    scaler = MinMaxScaler()
    scaler.fit(time_train_)
    feature_normalization_1 = scaler.transform(time_test_)
    feature_normalization_1 = feature_normalization_1.reshape(batch_test, 6, 1)
    test_set = np.concatenate((test_set[:, :, 1:], feature_normalization_1), axis=2)
    time_step = 6
    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    train_set.epoch_completed = 0
    previous_visit = 3
    predicted_visit = 3

    batch_size = 64
    epochs = 1

    # hidden_size = 2**(int(hidden_size))
    # z_dims = 2 ** (int(z_dims))
    # n_disc = int(n_disc)
    # learning_rate = 10**learning_rate
    # l2_regularization = 10**l2_regularization
    # imbalance_kl = 10 ** imbalance_kl
    # t_imbalance = 10 ** t_imbalance

    print('----batch_size{}---hidden_size{}---n_disc{}---epochs{}---'
          '---learning_rate{}---l2_regularization{}--kl_imbalance{}---z_dims{}----t_imbalance{}--'
          .format(batch_size, hidden_size, n_disc, epochs, learning_rate, l2_regularization,
                  imbalance_kl, z_dims, t_imbalance))

    generator = Decoder(feature_dims=feature_dims,
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
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape, tf.GradientTape() as encode_tape:
        l = 0
        while train_set.epoch_completed < epochs:
            input_x_train = train_set.next_batch(batch_size)
            if input_x_train is not None:
                input_x_train_feature = tf.cast(input_x_train[:, :, 1:], tf.float32)
                input_t_train = input_x_train[:, :, 0]
                # input_x_test = test_set.dynamic_features
                # input_x_test = tf.cast(input_x_test, tf.float32)[:, :, 1:]
                # input_t_test = input_x_test[:, :, 0]

                context_state = encode_context(input_x_train_feature)
                h_i = tf.reshape(context_state[:, -1, :], [-1, hidden_size])
                fake_input, mean_all_post, log_var_all_post, mean_all_prior, log_var_all_prior, z_all, time_estimates_probability_all = generator([h_i, input_t_train])

                l += 1
                print('第{}批次的数据训练'.format(l))
                mae_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train_feature[:, previous_visit:previous_visit+predicted_visit, :], fake_input))
                time_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(time_estimates_probability_all, 1e-15, 1.0)))
                print('time_loss----------{}'.format(time_loss))
                KL = tf.keras.losses.KLDivergence()
                for m in range(len(z_all)):
                    posterior_d = z_all[m][0]
                    prior_d = z_all[m][1]
                    kl_loss = - KL(posterior_d, prior_d)
                kl_loss_all = tf.reduce_mean(kl_loss)
                # kl_loss_all = []
                # for t_index in range(predicted_visit):
                #     for z_index in range(z_dims):
                #         log_var_prior = log_var_all_prior[t_index, z_index]
                #         log_var_post = log_var_all_post[t_index, z_index]
                #         mean_post  = mean_all_post[t_index, z_index]
                #         mean_prior = mean_all_prior[t_index, z_index]
                #         kld = 2 * tf.math.log(log_var_prior) - 2*tf.math.log(log_var_post) + \
                #               (tf.math.pow(log_var_post, 2) +
                #                tf.math.pow((mean_post - mean_prior), 2)) / tf.math.pow(log_var_prior, 2) - 1
                #         kl_loss_all.append(kld)
                # kl_loss_all = 0.5 ** tf.reduce_sum(kl_loss_all)
                gen_loss = mae_loss + kl_loss_all * imbalance_kl + time_loss*t_imbalance

                for weight in generator.trainable_variables:
                    gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                for weight in encode_context.trainable_variables:
                    gen_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                variables = [var for var in generator.trainable_variables]
                for var in encode_context.trainable_variables:
                    variables.append(var)

                gradient_generator = gen_tape.gradient(gen_loss, variables)
                generator_optimizer.apply_gradients(zip(gradient_generator, variables))

        print('开始测试！')
        input_x_test_all = tf.zeros(shape=(0, 6, feature_dims))
        generated_x_test_all = tf.zeros(shape=(0, predicted_visit, feature_dims))
        while test_set.epoch_completed < epochs:
            input_test = test_set.next_batch(batch_size)
            if input_test is not None:
                input_x_test = input_test[:, :, 1:]
                input_t_test = input_test[:, :, 0]
                context_state_test = encode_context(input_x_test)
                h_i_test = tf.reshape(context_state_test[:, -1, :], [-1, hidden_size])
                fake_input_test, mean_all_post_test, log_var_all_test_post, mean_all_test_prior, log_var_all_test_prior, z_all, time_estimates_probability_test = generator([h_i_test, input_t_test])
                generated_x_test_all = tf.concat((generated_x_test_all, fake_input_test), axis=0)
                input_x_test_all = tf.concat((input_x_test_all, input_x_test), axis=0)

        mse_loss_ = tf.reduce_mean(tf.keras.losses.mse(input_x_test_all[:, previous_visit:previous_visit+predicted_visit, :], generated_x_test_all))
        print("mse_loss:{}---------".format(mse_loss_))
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
        tf.compat.v1.reset_default_graph()
        return -1*mse_loss_
        # return -1*mse_loss_, np.mean(r_value_all)


def test_test(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

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
    test_test('HF_test_3_3_测试_7_10')
    # GAN_LSTM_BO = BayesianOptimization(
    #     train_step, {
    #         'hidden_size': (4, 7),
    #         'z_dims': (4, 7),
    #         'n_disc': (1, 20),
    #         'imbalance_kl': (-6, 0),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #         't_imbalance': (-5, -1),
    #     }
    # )
    # GAN_LSTM_BO.maximize()
    # print(GAN_LSTM_BO.max)
    # 心衰数据集
    mse_all = []
    for i in range(50):
        mse = train_step(hidden_size=16, n_disc=16, learning_rate=0.007472646519640033, l2_regularization=4.0849237295283014e-05,imbalance_kl=1.1286624308367073e-05, z_dims=32, t_imbalance=0.0008249623190603408)
        mse_all.append(mse)
        print('第{}次测试完成'.format(i))
    print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    print('----------------mse_std:{}----------'.format(np.std(mse_all)))

    # 青光眼数据
    # mse_all = []
    # for i in range(50):
    #     mse = train_step(hidden_size=64, n_disc=16, learning_rate=0.004008587109539563, l2_regularization=0.0007415250362469344, imbalance_kl=1.8120003016994191e-06, z_dims=16, t_imbalance=0.04648735776157675)
    #     mse_all.append(mse)
    #     print('第{}次测试完成'.format(i))
    # print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
    # print('----------------mse_std:{}----------'.format(np.std(mse_all)))

   # MIMIC 数据
   #  mse_all = []
   #  for i in range(50):
   #      mse = train_step(hidden_size=16, n_disc=10, learning_rate=0.010185994300925677, l2_regularization=0.00011018825517104857,imbalance_kl=1.561967029411175e-06, z_dims=32, t_imbalance=0.0042670789630865425)
   #      mse_all.append(mse)
   #      print('第{}次测试完成'.format(i))
   #  print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
   #  print('----------------mse_std:{}----------'.format(np.std(mse_all)))