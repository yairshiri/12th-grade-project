from collections import defaultdict
import numpy as np
from EnemyBrain.BaseClasses import Agent
import tensorflow as tf
import pickle
from EnemyBrain import Memory
from utils import get_var, safe_load, time_me
import tensorflow_probability as tfp


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


# DQN agent
class DQN(Agent):
    def _build_model(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.BatchNormalization())

        # adding the layers in the amounts specified
        for amount in self.hyperparameters['model']['layer sizes']:
            model.add(tf.keras.layers.Dense(amount, activation='selu',
                                            kernel_initializer=tf.keras.initializers.RandomNormal()))
            model.add(tf.keras.layers.BatchNormalization())

        # the output layer
        model.add(tf.keras.layers.Dense(self.num_of_actions, activation='linear'))
        model.build(input_shape=(1, self.env.observation_space))
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
        return self.policy.select_action(q_vals=self.predict(state))

    def predict(self, states):
        return self.model(np.atleast_2d(states))

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

        model.add(tf.keras.layers.BatchNormalization())

        # adding the layers in the amounts specified
        for amount in self.hyperparameters['model']['layer sizes']:
            model.add(tf.keras.layers.Dense(amount, activation='selu',
                                            kernel_initializer=tf.keras.initializers.RandomNormal()))
            model.add(tf.keras.layers.BatchNormalization())

        # the output layer
        model.add(tf.keras.layers.Dense(self.num_of_actions, activation='softmax'))
        model.build(input_shape=(1, self.env.observation_space))
        return model

    def __init__(self, env):
        super(PG, self).__init__(env)
        self.gamma = self.hyperparameters['hyperparameters']['gamma']
        self.alpha = self.hyperparameters['hyperparameters']['alpha']
        self.num_of_actions = self.hyperparameters['number of actions']
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        self.model = self._build_model()
        self.memory = {'rewards': [], 'log probs': [], 'states': [], 'actions': []}
        self.tape = tf.GradientTape()

    # resetting the memory
    def reset(self):
        self.memory = {'rewards': [], 'log probs': [], 'states': [], 'actions': []}
        self.tape = tf.GradientTape()

    def learn(self, state, action, reward, next_state, done=None):
        self.memory['rewards'].append(reward)
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        if done:
            episode_len = len(self.memory['rewards'])
            discounted_Rs = [self.memory['rewards'][t] * self.gamma ** t for t in range(episode_len)]
            sums = [np.sum(discounted_Rs[i:]) for i in range(episode_len)]
            # getting the loss
            with tf.GradientTape() as tape:
                logs_p = tf.math.reduce_sum(self.model(np.atleast_2d(self.memory['states']))* tf.keras.utils.to_categorical(self.memory['actions'], self.num_of_actions),axis=1)
                loss = tf.math.multiply(logs_p,sums)
            # calcing and applying gradiants
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.reset()

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
        print(p)
        p = np.array(p)
        action = np.random.choice(self.num_of_actions, p=p[0])
        self.memory['log probs'].append(dist.log_prob(action))
        return action

    # def get_loss(self, state, action):
    #     p = self.model(np.atleast_2d(state))
    #     dist = tfp.distributions.Categorical(probs=p)
    #     return dist.log_prob(action)
