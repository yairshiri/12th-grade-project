import random

import numpy as np
from EnemyBrain.BaseClasses import Memory

import numpy


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = {'s': [], 'a': [], 'r': [],
                     's2': [], 'done': []}
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = len(self.data['s']) + self.capacity - 1

        s, a, r, s2, d = data
        self.data['s'].append(s)
        self.data['a'].append(a)
        self.data['r'].append(r)
        self.data['s2'].append(s2)
        self.data['done'].append(d)
        self.update(idx, p)

        if len(self.data['s']) >= self.capacity:
            self.data['s'].pop(0)
            self.data['a'].pop(0)
            self.data['r'].pop(0)
            self.data['s2'].pop(0)
            self.data['done'].pop(0)

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self._get_sample(dataIdx)

    def _get_sample(self, idx):
        return self.data['s'][idx],self.data['a'][idx],self.data['r'][idx],self.data['s2'][idx],self.data['done'][idx]


class Vanilla_Memory(Memory):
    def __init__(self, min_size, max_size, batch_size):
        super().__init__(min_size, max_size, batch_size)

    def get_ids(self):
        ids = np.random.randint(low=0, high=len(self.memory['s']),
                                size=min(self.batch_size, len(self.memory['s'])))
        return ids

    def add_exp(self, state, action, reward, next_state, done):
        if len(self.memory['s']) == self.max_size:
            for key in self.memory.keys():
                self.memory[key].pop(0)
        self.memory['s'].append(state)
        self.memory['a'].append(action)
        self.memory['r'].append(reward)
        self.memory['s2'].append(next_state)
        self.memory['done'].append(done)

    def get_batch(self):
        ids = self.get_ids()
        dones = np.asarray([self.memory['done'][i] for i in ids])
        states = np.asarray([self.memory['s'][i] for i in ids])
        actions = np.asarray([self.memory['a'][i] for i in ids])
        rewards = np.asarray([self.memory['r'][i] for i in ids])
        states_next = np.asarray([self.memory['s2'][i] for i in ids])
        return states, actions, rewards, states_next, dones


class Prioritized_Memory(Memory):
    e = 0.01
    a = 0.6
    b = 0.4
    b_increment = 0.001

    def __init__(self, min_size, max_size, batch_size):
        super().__init__(min_size, max_size, batch_size)
        self.tree = SumTree(max_size)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def is_ready(self):
        return len(self.tree.data['s']) >= self.min_size

    def add_exp(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def get_batch(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.b = np.min([1., self.b + self.b_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.b)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
