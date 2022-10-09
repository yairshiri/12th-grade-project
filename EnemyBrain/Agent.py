from collections import defaultdict

import keras.layers
import numpy as np
from EnemyBrain.BaseClasses import Agent
import tensorflow as tf
import pickle
from EnemyBrain import Memory
from utils import get_var, safe_load, time_me
import tensorflow_probability as tfp
from tensorflow.keras import layers


# Q-learning agent
class Q(Agent):
    def __init__(self, env):
        super(Q, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.Q = defaultdict(lambda: np.zeros(self.hyperparameters['number of actions']))  # The Q-TABLE
        if self.mode == 'testing':
            self.load()

    # choosing the best possible action with a probability of 1-epsilon, or random action
    def action(self, state):
        return self.policy.select_action(self.Q[state])

    # use the formula to learn
    def learn(self, state, action, reward, next_state, **kwargs):
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    def save(self):
        name = get_var("Choose saving name")
        if name == "":
            return
        policy = {}
        for key, value in self.Q.items():
            policy[key] = self.Q[key]
        with open(f"{name}.pickle", 'bw') as f:
            pickle.dump(policy, f)

    @safe_load(f_type="pickle")
    def load(self, name):
        with open(f"{name}.pickle", "rb") as f:
            policy = pickle.load(f)
            if policy is None:
                return
            for key, val in policy.items():
                self.Q[key] = policy[key]
        self.env.instance.info['Loaded brain'] = name


# SARSA agent
class SARSA(Q):
    def __init__(self, env):
        super(SARSA, self).__init__(env)

    # use the formula to learn
    def learn(self, state, action, reward, next_state, **kwargs):
        self.Q[state][action] += self.alpha * (
                reward + self.gamma * self.Q[next_state][self.action(next_state)] - self.Q[state][action])


class XYTOintLayer(keras.layers.Layer):
    def __init__(self):
        super(XYTOintLayer, self).__init__()

    def call(self, inputs, **kwargs):
        xmove = tf.math.argmax(inputs[0])
        ymove = tf.math.argmax(inputs[1])
        return xmove, ymove


# DQN agent
class DQN(Agent):
    def _build_model(self):
        inputs = tf.keras.Input(self.env.observation_space)
        # if we want to use a conv net, use a convnet. otherwise use the specified network
        if self.env.instance.config['environment']['use conv']:
            layer = layers.Conv2D(32, 3, activation="relu")
            x = layers.TimeDistributed(layer)(inputs)
            layer = layers.Conv2D(32, 3, activation="relu")
            x = layers.TimeDistributed(layer)(x)
            layer = layers.Conv2D(32, 3, activation="relu")
            x = layers.TimeDistributed(layer)(x)
            layer = layers.MaxPool2D((2, 2))
            x = layers.TimeDistributed(layer)(x)
            layer = layers.Flatten()
            x = layers.TimeDistributed(layer)(x)
            x = layers.LSTM(64, return_sequences=False)(x)
            x = layers.Dense(100)(x)
        else:
            x = layers.LSTM(100, return_sequences=False)(inputs)
            # layer = tf.keras.layers.Dense(self.hyperparameters['model']['layer sizes'][0], activation='relu')
            # x = layers.TimeDistributed(layer)(inputs)
            for amount in self.hyperparameters['model']['layer sizes']:
                layer = tf.keras.layers.Dense(amount, activation='relu')
                x = layer(x)

        # the output layer
        outputs = tf.keras.layers.Dense(self.num_of_actions, activation='linear')(x)
        # movey = tf.keras.layers.Dense(self.num_of_actions, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        return model

    def __init__(self, env):
        super(DQN, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        batch_size = self.hyperparameters['replay buffer']['batch size']
        min_buffer_size = self.hyperparameters['replay buffer']['min size']
        max_buffer_size = self.hyperparameters['replay buffer']['max size']
        assert self.hyperparameters['replay buffer']['type'].lower() in ['per', 'uer']
        if self.hyperparameters['replay buffer']['type'].lower() == 'per':
            self.memory = Memory.Prioritized(min_buffer_size, max_buffer_size, batch_size)
        elif self.hyperparameters['replay buffer']['type'].lower() == 'uer':
            self.memory = Memory.Uniform(min_buffer_size, max_buffer_size, batch_size)
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        self.num_of_actions = self.hyperparameters['number of actions']
        self.learn_interval = self.hyperparameters['model']['learn interval']
        self.memory_mode = 'PER' if isinstance(self.memory, Memory.Prioritized) else 'UER'
        self.model = self._build_model()
        if self.mode == 'testing':
            self.load()

    def learn(self, state, action, reward, next_state, done=None):
        if self.memory_mode == 'PER':
            error = self.prepare_exp((state, action, reward, next_state, done))
            self.memory.add_exp(error, (state, action, reward, next_state, done))
        else:
            self.memory.add_exp(state, action, reward, next_state, done)
        if not self.memory.is_ready():
            return
        if self.env.latest_data['number of steps'] % self.learn_interval == 0:
            if self.memory_mode == 'PER':
                exp, idx, is_weights = self.memory.get_batch()
                states, actions, rewards, states_next, dones = zip(*exp)
            else:
                states, actions, rewards, states_next, dones = self.memory.get_batch()
            value_next = np.nanmax(self.predict(states_next), axis=1)
            actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)
            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.predict(states) * tf.keras.utils.to_categorical(actions, self.num_of_actions),
                    axis=1)
                if self.memory_mode == 'PER':
                    loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values) * is_weights)
                else:
                    loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            if self.memory_mode == 'PER':
                errors = np.abs(actual_values - selected_action_values)
                for i in range(len(states)):
                    self.memory.update(idx[i], errors[i])
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

    def prepare_exp(self, exp):
        states, actions, rewards, states_next, dones = exp
        value_next = np.nanmax(self.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)
        pred = self.predict(states)
        return np.abs(actual_values - pred[0][actions])[0]

    def action(self, state):
        qvals = self.predict(state)
        return self.policy.select_action(q_vals=qvals)

    def predict(self, states):
        if not isinstance(states, np.ndarray):
            states = np.array(states)
        if states.ndim == len(self.env.observation_space):
            states = np.array([states])
        return self.model(states)

    @safe_load(f_type="h5")
    def load(self, name):
        self.model.load_weights(f"{name}.h5")
        self.env.instance.info['Loaded brain'] = name

    def save(self):
        name = get_var("Choose saving name")
        if name == "":
            return
        self.model.save_weights(f"{name}.h5")


# Policy gradiant agent
class PG(Agent):
    def _build_model(self):
        model = tf.keras.models.Sequential()

        # adding the layers in the amounts specified
        for amount in self.hyperparameters['model']['layer sizes']:
            model.add(tf.keras.layers.Dense(amount, activation='selu',
                                            kernel_initializer=tf.keras.initializers.lecun_normal()))

        # the output layer
        model.add(tf.keras.layers.Dense(self.num_of_actions, activation='softmax'))
        model.compile(tf.keras.optimizers.Adam(self.alpha), 'categorical_crossentropy')
        return model

    def __init__(self, env):
        super(PG, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.num_of_actions = self.hyperparameters['number of actions']
        self.model = self._build_model()
        self.memory = {'rewards': [], 'log probs': [], 'states': [], 'actions': [], 'grads': [], 'probs': []}
        self.tape = tf.GradientTape()

    # resetting the memory
    def reset(self):
        self.memory = {'rewards': [], 'log probs': [], 'states': [], 'actions': [], 'grads': [], 'probs': []}
        self.tape = tf.GradientTape()

    def learn(self, state, action, reward, next_state, done=None):
        self.memory['rewards'].append(reward)
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        if done:
            episode_len = len(self.memory['rewards'])
            # discounted_Rs = [self.memory['rewards'][t] * self.gamma ** t for t in range(episode_len)]
            discounted_Rs = self.discount_rewards(self.memory['rewards'])
            # getting the loss
            # logs_p = tf.math.reduce_sum(self.model(np.atleast_2d(self.memory['states']))* tf.keras.utils.to_categorical(self.memory['actions'], self.num_of_actions),axis=1)
            # loss = -tf.math.multiply(logs_p,discounted_Rs)
            # calcing and applying gradiants
            grads = np.vstack(self.memory['grads'])
            grads = [grads[i] * discounted_Rs[i] for i in range(episode_len)]
            grads = self.alpha * np.vstack([grads]) + self.memory['probs']
            stets = np.vstack(self.memory['states'])
            self.model.train_on_batch(np.atleast_2d(self.memory['states']), grads)
            self.reset()

    def discount_rewards(self, rewards):
        discounted_rewards = []
        cumulative_total_return = 0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards) / (std_rewards + 1e-7)  # avoiding zero div

        return norm_discounted_rewards

    # def learn(self, state, action, reward, next_state, done=None):
    #     self.memory['rewards'].append(reward)
    #     self.memory['states'].append(state)
    #     self.memory['actions'].append(action)
    #     if done:
    #         episode_len = len(self.memory['rewards'])
    #         discounted_Rs = [self.memory['rewards'][t] * self.gamma ** t for t in range(episode_len)]
    #         sums = [np.sum(discounted_Rs[i:]) for i in range(episode_len)][::-1]
    #         # getting the loss
    #         with self.tape:
    #             # calcing the discounted loss
    #             loss = tf.reduce_sum([-self.memory['log probs'][i]*sums[i] for i in range(episode_len)])
    #         # calcing and applying gradiants
    #         grads = self.tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
    #         self.reset()

    def action(self, state):
        p = self.model(np.atleast_2d(state))
        dist = tfp.distributions.Categorical(probs=p)
        p = np.array(p)[0]
        # action = np.random.choice(self.num_of_actions, p=p[0])
        action = self.policy.select_action(q_vals=p)
        self.memory['log probs'].append(dist.log_prob(action))
        self.memory['probs'].append(p)
        grad = np.zeros(self.num_of_actions)
        grad[action] = 1
        grad -= p / np.sum(p)
        self.memory['grads'].append(grad)
        return action

    # def get_loss(self, state, action):
    #     p = self.model(np.atleast_2d(state))
    #     dist = tfp.distributions.Categorical(probs=p)
    #     return dist.log_prob(action)
