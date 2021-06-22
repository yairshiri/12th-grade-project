import numpy as np
from utils import boltzmann


class EpsGreedy:
    def __init__(self, num_actions, decay=0.9999, min_value=0.1, start_value=1.0, *args, **kwargs):
        self.num_actions = num_actions
        self.start_value = start_value
        self.min_value = max(min_value, 0.001)
        self.decay = decay
        self.epsilon = self.start_value

    def _update_eps(self):
        self.epsilon = max(self.epsilon * self.decay, self.min_value)
        return self.epsilon

    def select_action(self, q_vals):
        self._update_eps()
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(q_vals)

    def get_data(self):
        return {'epsilon': self.epsilon}


class EpsGreedyBoltzmann(EpsGreedy):
    def __init__(self, num_actions, decay=0.9999, min_value=0.1, start_value=1.0, *args, **kwargs):
        super().__init__(num_actions, decay, min_value, start_value, *args, **kwargs)

    def select_action(self, q_vals):
        self._update_eps()
        if np.random.random() < self.epsilon:
            p = boltzmann(q_vals,self.epsilon)
            print(q_vals)
            return np.random.choice(self.num_actions, p=p)
        else:
            return np.argmax(q_vals)


class Boltzmann(EpsGreedy):
    def __init__(self, num_actions, decay=0.9999, min_value=0.1, start_value=1.0, *args, **kwargs):
        super().__init__(num_actions, decay, min_value, start_value, *args, **kwargs)

    def select_action(self, q_vals):
        self._update_eps()
        p = boltzmann(q_vals,self.epsilon)
        return np.random.choice(self.num_actions,p=p)

    def get_data(self):
        return {'tau':self.epsilon}

