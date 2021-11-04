import networkx as nx
import numpy as np


def generate_modular_graph(nodes_in_mod=20, p_mod=0.2, connections=3):
    """Write some documentaion"""
    if nodes_in_mod is not None:
        if hasattr(nodes_in_mod, "__len__"):
            n1 = nodes_in_mod[0]
            n2 = nodes_in_mod[1]
        else:
            n1 = n2 = nodes_in_mod
        if n1 <= 0 or n2 <= 0:
            raise ValueError("number of nodes should be a positive number")
    else:
        raise ValueError("A very specific bad thing happened.")

    if p_mod is not None:
        if hasattr(p_mod, "__len__"):
            p1 = p_mod[0]
            p2 = p_mod[1]
        else:
            p1 = p2 = p_mod
        if p1 <= 0 or p2 <= 0:
            raise ValueError("number of nodes should be a positive number")
    else:
        raise ValueError("A very specific bad thing happened.")

    if connections is not None and connections > 0:
        conns = connections
    else:
        raise ValueError("A very specific bad thing happened.")

    g1 = nx.fast_gnp_random_graph(n1, p1, seed=None, directed=False)
    g2 = nx.fast_gnp_random_graph(n2, p2, seed=None, directed=False)
    g = nx.disjoint_union(g1, g2)
    r1 = np.random.choice(n1, conns, replace=False)
    r2 = np.random.choice(n2, conns, replace=False) + n1
    g.add_edges_from(zip(r1, r2))
    return g
