# import tensorflow_probability as tfp
from modify.data import DataSet
from TimeLSTMCell_3 import *
import scipy.stats as stats


# Encode the state into hidden state
class EncodeContext(tf.keras.layers.Layer):
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
            input_x = input_context[:, time, :]
            output, state = self.LSTM_Cell_encode(input_x, state)
            output = tf.reshape(output, [batch, 1, -1])
            context_state = np.concatenate((context_state, output), axis=1)
        return context_state


class Actor(tf.keras.layers.Layer):
    def __init__(self, n_output, hidden_size, batch_size, feature_dims):
        super().__init__(name='Actor')
        self.n_output = n_output
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.feature_dims = feature_dims
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)

    def call(self, state, feature_dims):
        action_var_all = tf.zeros(shape=(tf.shape(state)[0], 0))
        norm_dist_all = []
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        for i in range(feature_dims):
            mu = self.dense3(hidden_2)
            sigma = self.dense4(hidden_2)
            sigma = tf.nn.softplus(sigma) + 1e-5
            mu = tf.squeeze(mu)
            sigma = tf.squeeze(sigma)
            norm_dist = tf.compat.v1.distributions.Normal(mu, sigma)
            action_var = norm_dist.sample(1)
            action_var = tf.clip_by_value(action_var, 0, 1)
            action_var_all = tf.concat((action_var_all, tf.reshape(action_var, [-1, 1])), axis=1)
            norm_dist_all.append(norm_dist)
        return norm_dist_all, action_var_all


class CriticNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, n_output):
        super().__init__(name='Critic')
        self.hidden_size = hidden_size
        self.n_output = n_output
        self.dense1 = tf.keras.layers.Dense(units=hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size)
        self.dense3 = tf.keras.layers.Dense(units=n_output)

    def call(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        V = self.dense3(hidden_2)
        return V


def reward(current_action, input_x_current):
    # current_reward = tf.keras.losses.mse(current_action, input_x_current)
    current_action = np.squeeze(current_action)
    input_x_current = np.squeeze(input_x_current)
    reward_cal = []
    for batch_ in range(len(current_action)):
        current_reward = stats.wasserstein_distance(current_action[batch_], input_x_current[batch_])
        reward_cal.append(current_reward)

    # if -np.mean(reward_cal) > -0.15:
    #     return 1
    # else:
    #     return 0
    return -np.mean(reward_cal)


def train_step(hidden_size, learning_rate, l2_regularization, gamma):
    train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)
    time_step = 6
    feature_dims = train_set.shape[2]-1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)
    n_output = 1
    train_set.epoch_completed = 0
    batch_size = 64
    num_episode = 1000
    epochs = 1

    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    # gamma = 10 ** gamma

    print('----batch_size{}---hidden_size{}---num_episode{}------learning_rate{}---l2_regularization{}---'
          .format(batch_size, hidden_size, num_episode, learning_rate, l2_regularization))

    actor = Actor(n_output=n_output,
                  hidden_size=hidden_size,
                  batch_size=batch_size,
                  feature_dims=feature_dims)

    critic = CriticNetwork(hidden_size=hidden_size, n_output=n_output)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    with tf.GradientTape(persistent=True) as actor_tape, tf.GradientTape(persistent=True) as critic_tape:
        # while train_set.epoch_completed < epochs:
        # input_x_train = train_set.next_batch(batch_size)
        input_x_train = train_set.dynamic_features
        input_context = input_x_train[:, :, 1:]
        batch_ = tf.shape(input_context)[0]
        context_state = np.zeros(shape=[batch_, 0, hidden_size])
        LSTM_Cell_encode_layer = tf.keras.layers.LSTMCell(hidden_size)
        initial_state = None

        for time in range(time_step - 3):
            state = LSTM_Cell_encode_layer.get_initial_state(batch_size=batch_, dtype=tf.float32)
            input_x = input_context[:, time, :]
            output, state = LSTM_Cell_encode_layer(input_x, state)
            output = tf.reshape(output, [batch_, 1, -1])
            context_state = np.concatenate((context_state, output), axis=1)
            if time == 2:
                initial_state_ = state
        episode_history = []
        mse_all = []
        for episode in range(num_episode):
            # Encode the context state into state 0
            state = context_state[:, -1, :]  # initial state 0
            batch_now = tf.shape(state)[0]
            total_reward = 0
            action_all = np.zeros(shape=(batch_now, 0, feature_dims))
            done = False
            # initial_state = state
            for step in range(time_step-3):
                if step == 2:
                    done = True
                if step == 0:
                    initial_state = initial_state_
                input_x_current = input_x_train[:, step+3, 1:]
                norm_dist, current_action = actor(state, feature_dims)
                current_action = tf.squeeze(current_action)
                next_state, memory = LSTM_Cell_encode_layer(current_action, initial_state)
                action_all = np.concatenate((action_all, current_action.numpy().reshape(batch_now, 1, feature_dims)),
                                            axis=1)
                reward_ = reward(current_action, input_x_current)
                total_reward += reward_
                reward_ = tf.cast(reward_, tf.float32)
                v_of_next_state = tf.squeeze(critic(next_state))
                if done:
                    target = reward_
                else:
                    target = reward_ + gamma*v_of_next_state
                td_error = target - tf.squeeze(critic(state))

                critic_loss = tf.reduce_mean(tf.math.squared_difference(target, tf.squeeze(critic(state))))
                prob = 1
                for i in range(feature_dims):
                    action_ = current_action[:, i]
                    norm_dist_ = norm_dist[i]
                    prob *= tf.math.log(norm_dist_.prob(action_)+1e-5)

                actor_loss = -tf.reduce_mean(prob*td_error)

                for weight in critic.trainable_variables:
                    critic_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                for weight in actor.trainable_variables:
                    actor_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                gradient_actor = actor_tape.gradient(actor_loss, actor.trainable_variables)
                actor_optimizer.apply_gradients(zip(gradient_actor, actor.trainable_variables))

                gradient_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)
                critic_optimizer.apply_gradients(zip(gradient_critic, critic.trainable_variables))

                state = next_state
                initial_state = memory

            mse = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, 3:, 1:], action_all))
            mse_all.append(mse)
            total_reward_ = total_reward/(time_step-3)
            episode_history.append(tf.reduce_mean(total_reward_))
            print('---episode{}---mse{}---cumulative{}-----'.format(episode, mse, tf.reduce_mean(episode_history)))
    return np.mean(mse_all)


if __name__=='__main__':
    # GAN_time_LSTM_BO = BayesianOptimization(
    #         train_step, {
    #             'hidden_size': (4, 8),
    #             'learning_rate': (-5, 0),
    #             'l2_regularization': (-5, 0),
    #             'gamma': (-5, 0),
    #         }
    #     )
    # GAN_time_LSTM_BO.maximize(n_iter=30)
    # print(GAN_time_LSTM_BO.max)
    mean_reward = train_step(hidden_size=16, learning_rate=0.001, l2_regularization=0.0001, gamma=0.01)
    print('mean_reward_{}----------'.format(mean_reward))




