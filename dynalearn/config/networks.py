import networkx as nx
import numpy as np

from .config import Config
from .util import TransformConfig, WeightConfig


class NetworkConfig(Config):
    @classmethod
    def gnp(cls, num_nodes=1000, p=0.004, weights=None, transforms=None, layers=None):
        cls = cls()
        cls.name = "GNPNetworkGenerator"
        cls.num_nodes = num_nodes
        cls.p = p
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

        return cls

    @classmethod
    def gnm(cls, num_nodes=1000, m=2000, weights=None, transforms=None, layers=None):
        cls = cls()
        cls.name = "GNMNetworkGenerator"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms

        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

        return cls

    @classmethod
    def ba(cls, num_nodes=1000, m=2, weights=None, transforms=None, layers=None):
        cls = cls()
        cls.name = "BANetworkGenerator"
        cls.num_nodes = num_nodes
        cls.m = m
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms
        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers

        return cls

    @classmethod
    def mod_net(
        cls,
        nodes_in_mod=20,
        p_mod=0.2,
        connections=3,
        weights=None,
        transforms=None,
        layers=None,
    ):
        """Write some documentaion"""
        cls = cls()
        cls.name = "ModuleNetworkGenerator"
        if isinstance(nodes_in_mod, list) and len(nodes_in_mod) >=2:
            cls.num_nodes = np.sum(nodes_in_mod)
            cls.first_node_size = nodes_in_mod[0]
        else:
            cls.num_nodes = nodes_in_mod * 2
            cls.first_node_size = nodes_in_mod

        cls.nodes_in_mod = nodes_in_mod
        cls.p_mod = p_mod
        cls.connections = connections
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms
        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers
        return cls

    @classmethod
    def w_gnp(cls, num_nodes=1000, p=0.004):
        w = WeightConfig.uniform()
        t = TransformConfig.sparcifier()
        cls = cls.gnp(num_nodes=num_nodes, p=p, weights=w, transforms=t)
        return cls

    @classmethod
    def w_ba(cls, num_nodes=1000, m=2):
        w = WeightConfig.uniform()
        t = TransformConfig.sparcifier()
        cls = cls.ba(num_nodes=num_nodes, m=m, weights=w, transforms=t)
        return cls

    @classmethod
    def mw_ba(cls, num_nodes=1000, m=2, layers=1):
        w = WeightConfig.uniform()
        t = TransformConfig.sparcifier()
        cls = cls.ba(num_nodes=num_nodes, m=m, weights=w, transforms=t, layers=layers)
        return cls

    @property
    def is_weighted(self):
        return "weights" in self.__dict__

    @property
    def is_multiplex(self):
        return "layers" in self.__dict__
