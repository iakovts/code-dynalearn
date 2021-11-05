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


def transform_mod_args(arg):
    if isinstance(arg, str):
        raise ValueError(f"Use float, int or an iterable")
    try:
        iter(arg)
    except TypeError:
        # If only a single float and int are passed for n_mod and p_mod,
        # Use them to create two identical modules
        arg = [arg] * 2
    return arg

def randm(i, mod_number):
    """Returns the index of a random module to connect to, excluding the current"""
    j = np.random.choice(mod_number, replace=True)
    while i == j:
        j = np.random.choice(mod_number, replace=True)
    return j


def gen_modular_network(n_mod=40, p_mod=0.2, connections=3):
    """Creates a modular network
    - args:
        :n_mod -> int or iterable: Number of nodes per module
        :p_mod -> float or iterable: edge probability within module (0,1)
        :connections -> int or iterable: Number of connections to other modules
    """
    n_mod = transform_mod_args(n_mod)
    p_mod = transform_mod_args(p_mod)

    if len(n_mod) != len(p_mod):
        raise ValueError("`p_mod` and `n_mod` should be vectors of the same length.")
    if connections > np.min(n_mod):
        raise ValueError(
            "Number of `connections` must be lower or equal to the number of nodes per module"
        )
    elif connections <= 0:
        raise ValueError("Connections must be greater than 0")

    g_tot = [
        nx.fast_gnp_random_graph(n_mod[i], p_mod[i], seed=None, directed=None)
        for i in range(len(p_mod))
    ]
    g = nx.disjoint_union_all(g_tot)

    conn_edges = []
    mod_number = range(len(g_tot))  # Number of modules
    print(mod_number)
    print(g_tot)
    for i in mod_number:
        for _ in range(connections):
            j = randm(i, mod_number)
            n1 = np.random.choice(n_mod[i], replace=False) + np.sum(n_mod[0:i-1])
            n2 = np.random.choice(n_mod[j], replace=False) + np.sum(n_mod[0:j-1])
            print(n1, n2)
            g.add_edge(n1, n2)

    return (g, g_tot)
