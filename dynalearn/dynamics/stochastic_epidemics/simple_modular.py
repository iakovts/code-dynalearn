import numpy as np

from .base import StochasticEpidemics
from dynalearn.dynamics.activation import independent
from dynalearn.config import Config


class SISModular(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        super().__init__(config, 2)
        self.infection = config.infection
        self.recovery = config.recovery

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        ltp = np.zeros((x.shape[0], self.num_states))
        p = independent(self.neighbors_state(x)[1], self.infection)
        q = self.recovery
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 1, 0] = q
        ltp[x == 1, 1] = 1 - q
        return ltp

    def number_of_infected(self, x):
        return np.sum(x == 1)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 1
        return x

    def initial_state(self, init_param=None, squeeze=True):
        if init_param is None:
            init_param = self.init_param
        if init_param is None:
            init_param = np.random.rand(self.num_states)
            init_param /= init_param.sum()
        elif isinstance(init_param, list):
            init_param = np.array(init_param)

        if not isinstance(init_param, np.ndarray):
            raise TypeError("`init_param` should be an numpy.ndarray")
        if not init_param.shape == (self.num_states,):
            raise ValueError("init_param and num_states mustshould have the same length")
        x = np.zeros((self.num_nodes, 1))
        affected = np.random.multinomial(1, init_param, self.first_node_size)
        affected = np.where(affected == 1.0)[1]
        affected = affected.reshape(*affected.shape, 1).repeat(self.lag, -1)
        x[:self.first_node_size] = affected
        if squeeze:
            return x.squeeze()
        else:
            return x
