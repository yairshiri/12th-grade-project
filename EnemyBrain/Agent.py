from collections import defaultdict
import numpy as np
import random
from EnemyBrain.BaseClasses import Agent
import tensorflow as tf
import pickle
import os
from EnemyBrain import Memory


class Q_Agent(Agent):
    def __init__(self, env):
        super(Q_Agent, self).__init__(env)
        self.eps_start = self.hyperparameters['hyperparameters']['epsilon']['starting']
        self.eps_decay = self.hyperparameters['hyperparameters']['epsilon']['decay']
        self.eps_min = self.hyperparameters['hyperparameters']['epsilon']['min']
        self.eps = self.eps_start
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.nb_actions = self.hyperparameters['number of actions']
        self.Q = defaultdict(lambda: np.zeros(self.nb_actions))  # The Q-TABLE
        if self.mode == 'testing':
            self.load()

    # choosing the best possible action with a probability of 1-epsilon, or random action
    def action(self, state):
        if self.mode == 'training':
            self.update_eps()
            prob = random.random()
            if prob > self.eps:
                return np.argmax(self.Q[state])
            else:
                return np.random.choice(np.arange(self.nb_actions))
        elif self.mode == 'testing':
            return np.argmax(self.Q[state])

    # use the formula to learn
    def learn(self, state, action, reward, next_state, **kwargs):
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    # updating the epsilon of the agent
    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        return self.eps

    def _save(self, name):
        if name == "":
            return
        policy = {}
        for key, value in self.Q.items():
            policy[key] = self.Q[key]
        with open(f"{name}.pickle", 'bw') as f:
            pickle.dump(policy, f)

    def _load(self, name):
        if name == "":
            return
        try:
            with open(f"{name}.pickle", "rb") as f:
                policy = pickle.load(f)
                if policy is None:
                    return
                for key, val in policy.items():
                    self.Q[key] = policy[key]
        except Exception as e:
            raise e


class SARSA_Agent(Q_Agent):
    def __init__(self, env):
        super(SARSA_Agent, self).__init__(env)

    # use the formula to learn
    def learn(self, state, action, reward, next_state, **kwargs):
        self.Q[state][action] += self.alpha * (
                reward + self.gamma * self.Q[next_state][self.action(next_state)] - self.Q[state][action])


class DQN_Agent(Agent):
    def _build_model(self):
        model = tf.keras.models.Sequential()
        # adding the layers in the amounts specified
        for amount in self.hyperparameters['model']['layer sizes']:
            model.add(tf.keras.layers.Dense(amount, activation='selu',
                                            kernel_initializer=tf.keras.initializers.RandomNormal()))
        # the output layer
        model.add(tf.keras.layers.Dense(self.num_of_actions, activation='linear'))
        model.build(input_shape=(1, self.env.observation_space))
        return model

    def __init__(self, env):
        super(DQN_Agent, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.batch_size = self.hyperparameters['replay buffer']['batch size']
        self.min_buffer_size = self.hyperparameters['replay buffer']['min size']
        self.max_buffer_size = self.hyperparameters['replay buffer']['max size']
        self.memory = Memory.Prioritized_Memory(self.min_buffer_size, self.max_buffer_size, self.batch_size) if \
        self.hyperparameters['replay buffer']['use prioritized'] else Memory.Uniform_replay(self.min_buffer_size,
                                                                                            self.max_buffer_size,
                                                                                            self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        self.eps_start = self.hyperparameters['hyperparameters']['epsilon']['starting']
        self.eps_decay = self.hyperparameters['hyperparameters']['epsilon']['decay']
        self.eps_min = self.hyperparameters['hyperparameters']['epsilon']['min']
        self.eps = self.eps_start
        self.num_of_actions = self.hyperparameters['number of actions']
        self.learn_interval = self.hyperparameters['model']['learn interval']
        self.memory_mode = 'PER' if isinstance(self.memory, Memory.Prioritized_Memory) else 'UER'
        self.model = self._build_model()
        if self.mode == 'testing':
            self.load()

    # updating the epsilon of the agent
    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        return self.eps

    def learn(self, state, action, reward, next_state, done=None):
        if self.memory_mode == 'PER':
            error = self.prepare_exp((state, action, reward, next_state, done))
            self.memory.add_exp(error, (state, action, reward, next_state, done))
        else:
            self.memory.add_exp(state, action, reward, next_state, done=None)
        if not self.memory.is_ready():
            return
        self.update_eps()
        if self.env.latest_data['number of steps'] % self.hyperparameters['model']['learn interval'] == 0:
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
                errors = tf.abs(actual_values - selected_action_values)
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
        loss = []
        for i in range(len(actual_values)):
            loss.append(np.mean(np.square(actual_values[i] - pred[i])))
        return loss

    def action(self, states):
        if self.mode == 'training':
            if np.random.random() < self.eps:
                return np.random.choice(self.num_of_actions)
            else:
                return np.argmax(self.predict(states))
        else:
            return np.argmax(self.predict(states))

    def predict(self, states):
        return self.model(np.atleast_2d(states))

    def _load(self, name):
        try:
            self.model.load_weights(f"{name}.h5")
        except Exception as e:
            pass

    def _save(self, name):
        self.model.save_weights(f"{name}.h5")


class DDQN_Agent(Agent):
    class QNET:
        def __init__(self, model):
            self.model = model

        def predict(self, states):
            return self.model(np.atleast_2d(states))

    def __init__(self, env):
        super(DDQN_Agent, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.batch_size = self.hyperparameters['replay buffer']['batch size']
        self.min_buffer_size = self.hyperparameters['replay buffer']['min size']
        self.max_buffer_size = self.hyperparameters['replay buffer']['max size']
        self.memory = Memory.Prioritized_Memory(self.min_buffer_size, self.max_buffer_size, self.batch_size)
        self.memory_mode = 'PER' if isinstance(self.memory, Memory.Prioritized_Memory) else 'UER'
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        self.eps_start = self.hyperparameters['hyperparameters']['epsilon']['starting']
        self.eps_decay = self.hyperparameters['hyperparameters']['epsilon']['decay']
        self.eps_min = self.hyperparameters['hyperparameters']['epsilon']['min']
        self.eps = self.eps_start
        self.num_of_actions = self.hyperparameters['number of actions']
        self.learn_interval = self.hyperparameters['model']['learn interval']
        self.target_net = DDQN_Agent.QNET(self._build_model())
        self.train_net = DDQN_Agent.QNET(self._build_model())
        if self.mode == 'testing':
            self.load()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        # the input layer
        # adding the layers in the amounts specified
        for amount in self.hyperparameters['model']['layer sizes']:
            model.add(tf.keras.layers.Dense(amount, activation='selu',
                                            kernel_initializer=tf.keras.initializers.RandomNormal()))
        # the output layer
        model.add(tf.keras.layers.Dense(self.num_of_actions, activation='linear'))
        # model.compile(tf.keras.optimizers.RMSprop(self.alpha), loss='mse')
        model.build(input_shape=(1, 4))
        return model

    # updating the epsilon of the agent
    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        return self.eps

    def copy_weights(self):
        self.target_net.model.set_weights(self.train_net.model.get_weights())

    # def learn(self, state, action, reward, next_state, done=None):
    #     self.memory.add_exp(state, action, reward, next_state, done)
    #     if not self.memory.is_ready():
    #         return
    #     self.update_eps()
    #
    #     if self.env.get_metric('number of steps') % self.hyperparameters['model']['copy interval'] == 0:
    #         self.copy_weights()
    #     if self.env.get_metric('number of steps') % self.hyperparameters['model']['learn interval'] == 0:
    #         states, actions, rewards, states_next, dones = self.memory.get_batch()
    #         value_next = self.target_net.predict(states_next)
    #         actual_values = rewards
    #         for i, val in enumerate(actual_values):
    #             if not dones[i]:
    #                 actual_values[i] += self.gamma * np.max(value_next[i])
    #         with tf.GradientTape() as tape:
    #             selected_action_values = tf.math.reduce_sum(
    #                 self.train_net.predict(states) * tf.keras.utils.to_categorical(actions, self.num_of_actions),
    #                 axis=1)
    #             loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
    #         variables = self.train_net.model.trainable_variables
    #         gradients = tape.gradient(loss, variables)
    #         self.optimizer.apply_gradients(zip(gradients, variables))

    def learn(self, state, action, reward, next_state, done=None):
        if self.memory_mode == 'PER':
            error = self.prepare_exp((state, action, reward, next_state, done))
            self.memory.add_exp(error, (state, action, reward, next_state, done))
        else:
            self.memory.add_exp(state, action, reward, next_state, done=None)
        if not self.memory.is_ready():
            return
        self.update_eps()
        if self.env.latest_data['number of steps'] % self.hyperparameters['model']['copy interval'] == 0:
            self.copy_weights()
        if self.env.latest_data['number of steps'] % self.hyperparameters['model']['learn interval'] == 0:
            if self.memory_mode == 'PER':
                exp, idx, is_weights = self.memory.get_batch()
                states, actions, rewards, states_next, dones = zip(*exp)
                if len(states) < self.batch_size:
                    print(1)
            else:
                states, actions, rewards, states_next, dones = self.memory.get_batch()
            value_next = np.nanmax(self.target_net.predict(states_next), axis=1)
            actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)
            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.train_net.predict(states) * tf.keras.utils.to_categorical(actions, self.num_of_actions),
                    axis=1)
                if self.memory_mode == 'PER':
                    loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values) * is_weights)
                else:
                    loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            if self.memory_mode == 'PER':
                errors = tf.abs(actual_values - selected_action_values)
                for i in range(len(states)):
                    self.memory.update(idx[i], errors[i])
            variables = self.train_net.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

    def prepare_exp(self, exp):
        states, actions, rewards, states_next, dones = exp
        value_next = np.nanmax(self.target_net.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)
        pred = self.train_net.predict(states)
        loss = []
        for i in range(len(actual_values)):
            loss.append(np.mean(np.square(actual_values[i] - pred[i])))
        return loss

    def action(self, states):
        if self.mode == 'training':
            if np.random.random() < self.eps:
                return np.random.choice(self.num_of_actions)
            else:
                return np.argmax(self.train_net.predict(states))
        else:
            return np.argmax(self.train_net.predict(states))

    def _load(self, name):
        try:
            self.train_net.model.load_weights(f"{name}.h5")
        except Exception as e:
            pass

    def _save(self, name):
        self.train_net.model.save_weights(f"{name}.h5")
