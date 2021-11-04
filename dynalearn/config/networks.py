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
    def mod_net(cls, nodes_in_mod=20, p_mod=0.2, connections=3, weights=None, transforms=None, layers=None):
        """Write some documentaion"""
        cls = cls()
        cls.name = "ModuleNetworkGenerator"
        cls.num_nodes = np.sum(nodes_in_mod)
        cls.nodes_in_mod = nodes_in_mod
        cls.p_mod = p_mod
        cls.connections = connections
        # cls.num_nodes = num_node
        if weights is not None:
            cls.weights = weights
        if transforms is not None:
            cls.transforms = transforms
        if isinstance(layers, int):
            cls.layers = [f"layer{i}" for i in range(layers)]
        elif isinstance(layers, list):
            cls.layers = layers
        return cls


    # @classmethod
    # def mod_net(
    #     cls,
    #     cliques=2,
    #     cliq_size=500,
    #     m=2,
    #     num_nodes=1000,
    #     transforms=None,
    #     weights=None,
    #     layers=None,
    # ):
    #     """Returns a modular network generator object"""
    #     cls = cls()
    #     cls.name = "ModuleNetworkGenerator"
    #     cls.num_nodes = num_nodes
    #     cls.m = m
    #     cls.cliques = cliques
    #     cls.cliq_size = cliq_size

    #     if weights is not None:
    #         cls.weights = weights
    #     if transforms is not None:
    #         cls.transforms = transforms

    #     if isinstance(layers, int):
    #         cls.layers = [f"layer{i}" for i in range(layers)]
    #     elif isinstance(layers, list):
    #         cls.layers = layers
    #     return cls

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


def generate_modular_graph_2(nodes_in_mod=20, p_mod=0.2, connections=3):
    """Write some documentaion"""
    try:
        if len(nodes_in_mod) >= 2:
            for n in nodes_in_mod:
                if n <= 0 or n < connections:
                    raise ValueError(
                        "Number of nodes should be greater than number of connections (and 0)."
                    )
        else:
            raise ValueError("At least 2 modules should exist.")
    except TypeError:  # catch TypeError if nodes_in_mod has not __len__ (aka is `int` or else)
        print("`nodes_in_mod` should be an iterable (list, tuple etc)")
        raise

    # if nodes_in_mod is not None:
    #     if hasattr(nodes_in_mod, "__len__"):
    #         n1 = nodes_in_mod[0]
    #         n2 = nodes_in_mod[1]
    #     else:
    #         n1 = n2 = nodes_in_mod
    #     if n1 <= 0 or n2 <= 0:
    #         raise ValueError("number of nodes should be a positive number")
    # else:
    #     raise ValueError("A very specific bad thing happened.")
    try:
        if len(p_mod) >= 2:
            for p in p_mod:
                if p < 0 or p > 1:
                    raise ValueError(
                        "Edge probability should be a positive number in (0, 1)."
                    )
        else:
            raise ValueError("Each module should have its probability")
    except TypeError:
        print("`p_mod` should be an iterable (list, tuple, etc)")

    if len(p_mod) != len(nodes_in_mod):
        raise ValueError("Length of `nodes_in_mod` should be equal to `p_mod`")

    # if p_mod is not None:
    #     if hasattr(p_mod, "__len__"):
    #         p1 = p_mod[0]
    #         p2 = p_mod[1]
    #     else:
    #         p1 = p2 = p_mod
    #     if p1 <= 0 or p2 <= 0:
    #         raise ValueError("number of nodes should be a positive number")
    # else:
    #     raise ValueError("A very specific bad thing happened.")

    if connections is not None and connections > 0:
        conns = connections
    else:
        raise ValueError("A very specific bad thing happened.")

    g_tot = [
        nx.fast_gnp_random_graph(nodes_in_mod[i], p_mod[i]) for i in range(len(p_mod))
    ]
    # g1 = nx.fast_gnp_random_graph(n1, p1, seed=None, directed=False)
    # g2 = nx.fast_gnp_random_graph(n2, p2, seed=None, directed=False)
    # g = nx.disjoint_union(g1, g2)
    g = nx.disjoint_union_all(g_tot)
    # r1 = np.random.choice(n1, conns, replace=False)
    # r2 = np.random.choice(n2, conns, replace=False) + n1
    g.add_edges_from(zip(r1, r2))
    return g
