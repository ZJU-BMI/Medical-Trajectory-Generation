import tensorflow as tf
import numpy as np
import time
import scipy.stats as stats
from data import DataSet


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
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        action = self.dense3(hidden_2)
        action = self.dense4(action)
        action = tf.clip_by_value(action, 0, 1)
        return action, action


class Critic(tf.keras.layers.Layer):
    def __init__(self, hidden_size, n_output):
        super().__init__(name='Critic')
        self.hidden_size = hidden_size
        self.n_output = n_output
        self.dense1 = tf.keras.layers.Dense(units=hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size)
        self.dense3 = tf.keras.layers.Dense(units=n_output)
        self.dense4 = tf.keras.layers.Dense(units=hidden_size)
        self.dense5 = tf.keras.layers.Dense(units=hidden_size)

    def call(self, state, a):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        hidden_3 = self.dense4(a)
        hidden_4 = self.dense5(hidden_3)
        hidden = tf.concat((hidden_2, hidden_4), axis=1)
        V = self.dense3(hidden)
        return V

    def gradients(self, states, actions):
        v = self.call(states, actions)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(actions)
            self.action_grads = tape.gradient(v, actions)
            del tape
        return self.action_grads


def reward(current_action, input_x_current):
    # current_reward = tf.keras.losses.mse(current_action, input_x_current)
    current_action = np.squeeze(current_action)
    input_x_current = np.squeeze(input_x_current)
    reward_cal = np.zeros(shape=(0, 1))
    for batch_ in range(len(current_action)):
        current_reward = stats.wasserstein_distance(current_action[batch_], input_x_current[batch_])
        # current_reward = tf.keras.losses.mse(current_action[batch_], input_x_current[batch_])
        reward_cal = np.concatenate((reward_cal, np.array(current_reward).reshape(-1, 1)))
    return -reward_cal


def train_step(actor_hidden_size, critic_hidden_size, LSTM_hidden_size, n_output,
               batch_size, memory_capacity, gamma, actor_lr, critic_lr, l2_actor, l2_critic):
    train_set = np.load('HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('HF_test_.npy').reshape(-1, 6, 30)
    time_step = 6
    feature_dims = train_set.shape[2] - 1

    train_set = DataSet(train_set)
    test_set = DataSet(test_set)

    actor_current = Actor(n_output, actor_hidden_size, batch_size, feature_dims)
    actor_target = Actor(n_output, actor_hidden_size, batch_size, feature_dims)
    critic_current = Critic(critic_hidden_size, n_output)
    critic_target = Critic(critic_hidden_size, n_output)

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    memory_network = np.zeros((memory_capacity, 2*LSTM_hidden_size+feature_dims+1), dtype=np.float32)
    pointer = 0
    num_episode = 1000
    epochs = 1
    tau = 0.001

    with tf.GradientTape(persistent=True) as actor_tape, tf.GradientTape(persistent=True) as critic_tape:
        # while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size)
        input_context = input_x_train[:, :, 1:]
        batch_ = tf.shape(input_context)[0]
        context_state = np.zeros(shape=[batch_, 0, LSTM_hidden_size])
        LSTM_Cell_encode_layer = tf.keras.layers.LSTMCell(LSTM_hidden_size)
        initial_state = None

        for time_ in range(time_step-3):
            state = LSTM_Cell_encode_layer.get_initial_state(batch_size=batch_, dtype=tf.float32)
            input_x = input_context[:, time_, :]
            output, state = LSTM_Cell_encode_layer(input_x, state)
            output = tf.reshape(output, [batch_, 1, -1])
            context_state = np.concatenate((context_state, output), axis=1)
            if time_ == 2:
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
            for step in range(time_step - 3):
                if step == 2:
                    done = True
                if step == 0:
                    initial_state = initial_state_
                input_x_current = input_x_train[:, step + 3, 1:]
                norm_dist, current_action = actor_current(state)
                current_action = tf.squeeze(current_action)
                next_state, memory = LSTM_Cell_encode_layer(current_action, initial_state)  # 模拟环境获得reward,next_state
                action_all = np.concatenate(
                    (action_all, current_action.numpy().reshape(batch_now, 1, feature_dims)), axis=1)
                reward_ = reward(current_action, input_x_current)
                total_reward += reward_
                reward_ = tf.cast(reward_, tf.float32)

                # memory 记忆库(若容量充足，则直接将原始数据push.若容量不充足，需要将原始的数据左移到指定的位置，再将数据push)
                memory_ = tf.concat((state, current_action, reward_, next_state), axis=1)
                index = pointer % memory_capacity
                if pointer+len(memory_network) <= memory_capacity:
                    memory_network[index:len(memory_)+index, :] = memory_
                    pointer += 1 * batch_size
                else:
                    memory_save = memory_network[pointer+batch_size-memory_capacity:pointer, :]
                    memory_network[:memory_capacity-batch_size, :] = memory_save
                    memory_network[memory_capacity-batch_size:, :] = memory_
                    pointer = memory_capacity

                if pointer >= memory_capacity:
                    softreplace = [tf.compat.v1.assign(t, (1 - tau) * t + e * tau)
                                   for t, e in
                                   zip(actor_target.trainable_variables + critic_target.trainable_variables,
                                       actor_current.trainable_variables + critic_current.trainable_variables)]

                    indices = np.random.choice(memory_capacity, size=batch_size)
                    bt = memory_network[indices, :]
                    bs = bt[:, :LSTM_hidden_size]
                    ba = bt[:, LSTM_hidden_size:LSTM_hidden_size+feature_dims]
                    br = bt[:, -LSTM_hidden_size-1:-LSTM_hidden_size]
                    bs_ = bt[:, -LSTM_hidden_size:]

                    if done:
                        target = br
                    else:
                        _, action_ = actor_target(bs_)
                        target = br + gamma * critic_target(bs_, action_)

                    # critic 参数更新
                    critic_loss = tf.reduce_mean(tf.math.squared_difference(target, critic_current(bs, ba)))
                    for weight in critic_current.trainable_variables:
                        critic_loss += tf.keras.regularizers.l2(l2_critic)(weight)

                    gradient_critic = critic_tape.gradient(critic_loss, critic_current.trainable_variables)
                    critic_optimizer.apply_gradients(zip(gradient_critic, critic_current.trainable_variables))
                    #  actor 参数更新
                    _, action_ = actor_current(bs)
                    actor_loss = -tf.reduce_mean(critic_current(bs, action_))
                    for weight in actor_current.trainable_variables:
                        actor_loss += tf.keras.regularizers.l2(l2_actor)(weight)
                    gradient_actor = actor_tape.gradient(actor_loss, actor_current.trainable_variables)
                    actor_optimizer.apply_gradients(zip(gradient_actor, actor_current.trainable_variables))

                state = next_state
                initial_state = memory
                total_reward_ = total_reward / (time_step - 3)
                episode_history.append(tf.reduce_mean(total_reward_))

            m = tf.keras.losses.mse(input_x_train[:, 3:, 1:], action_all)
            mse = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, 3:, 1:], action_all))
            mse_all.append(mse)
            print('---episode{}---mse{}---cumulative{}-----'.format(episode, mse, tf.reduce_mean(episode_history)))

    return np.mean(mse_all)


if __name__ == '__main__':
    m = train_step(actor_hidden_size=64, critic_hidden_size=64, LSTM_hidden_size=16, n_output=1,
                   batch_size=64, memory_capacity=100, gamma=0.01, actor_lr=0.001, critic_lr=0.001,
                   l2_actor=0.001, l2_critic=0.001)

    print(m)