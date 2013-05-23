#!/usr/bin/env python

"""A simple MDL calculator using the Map Equation and a graph."""

import argparse
from math import log
import networkx as nx
import sys

TAU = 0.15
PAGE_RANK = 'page_rank'
MODULE_ID = 'module_id'

def log2(prob):
    "Returns the log of prob in base 2"
    return log(prob, 2)

def entropy1(prob):
    """Half of the entropy function, as used in the InfoMap paper.
    entropy1(p) = p * log2(p)
    """
    if prob == 0:
        return 0
    return prob * log2(prob)

class Module:
    """Stores the information about a single module"""
    def __init__(self, module_id, nodes, graph):
        self.module_id = module_id
        self.nodes = frozenset(nodes)
        self.graph = graph
        self.prop_nodes = 1 - float(len(self.nodes)) / len(graph)
        # Set the module_id for every node
        for node in nodes:
            graph.node[node][MODULE_ID] = module_id
        # Compute the total PageRank
        self.total_pr = sum([graph.node[node][PAGE_RANK] for node in nodes])
        # Compute q_out, the exit probability of this module
        # .. Left half: tau * (n - n_i) / n * sum{alpha in i}(p_alpha)
        self.q_out = self.total_pr * TAU * self.prop_nodes
        # .. Right half: (1-tau) * sum{alpha in i}(sum{beta not in i}
        #                  p_alpha weight_alpha,beta)
        # This is what's in [RAB2009 eq. 6]. But it's apparently wrong if
        # node alpha has no out-edges, which is not in the paper.
        # ..
        # Implementing it with Seung-Hee's correction about dangling nodes
        for node in self.nodes:
            edges = graph.edges(node, data=True)
            page_rank = graph.node[node][PAGE_RANK]
            if len(edges) == 0:
                self.q_out += page_rank * self.prop_nodes * (1 - TAU)
                continue
            for (_, dest, data) in edges:
                if dest not in self.nodes:
                    self.q_out += page_rank * data['weight'] * (1 - TAU)
        self.q_plus_p = self.q_out + self.total_pr

    def get_codebook_length(self):
        "Computes module codebook length according to [RAB2009, eq. 3]"
        first = -entropy1(self.q_out / self.q_plus_p)
        second = -sum( \
                [entropy1(self.graph.node[node][PAGE_RANK]/self.q_plus_p) \
                    for node in self.nodes])
        return (self.q_plus_p) * (first + second)


class Clustering:
    "Stores a clustering of the graph into modules"
    def __init__(self, graph, modules):
        self.graph = graph
        self.total_pr_entropy = sum([entropy1(graph.node[node][PAGE_RANK]) \
                for node in graph])
        self.modules = [Module(module_id, module, graph) \
                for (module_id, module) in enumerate(modules)]

    def get_mdl(self):
        "Compute the MDL of this clustering according to [RAB2009, eq. 4]"
        total_qout = 0
        total_qout_entropy = 0
        total_both_entropy = 0
        for mod in self.modules:
            q_out = mod.q_out
            total_qout += q_out
            total_qout_entropy += entropy1(q_out)
            total_both_entropy += entropy1(mod.q_plus_p)
        term1 = entropy1(total_qout)
        term2 = -2 * total_qout_entropy
        term3 = -self.total_pr_entropy
        term4 = total_both_entropy
        return term1 + term2 + term3 + term4

    def get_index_codelength(self):
        "Compute the index codebook length according to [RAB2009, eq. 2]"
        if len(self.modules) == 1:
            return 0
        total_q = sum([mod.q_out for mod in self.modules])
        entropy = -sum([entropy1(mod.q_out / total_q) for mod in self.modules])
        return total_q * entropy

    def get_module_codelength(self):
        "Compute the module codebook length according to [RAB2009, eq. 3]"
        return sum([mod.get_codebook_length() for mod in self.modules])

def print_tree_file(graph, modules):
    """Produces a .tree file from the given clustering that is compatible with
    the InfoMapCheck utility."""
    for (mod_id, mod) in enumerate(modules):
        for (node_id, node) in enumerate(mod):
            print "%d:%d %f \"%s\"" % (mod_id+1, node_id+1,
                    graph.node[node][PAGE_RANK], node)

def load_and_process_graph(filename):
    """Load the graph, normalize edge weights, compute pagerank, and store all
    this back in node data."""
    # Load the graph
    graph = nx.DiGraph(nx.read_pajek(filename))
    print "Loaded a graph (%d nodes, %d edges)" % (len(graph),
            len(graph.edges()))
    # Compute the normalized edge weights
    for node in graph:
        edges = graph.edges(node, data=True)
        total_weight = sum([data['weight'] for (_, _, data) in edges])
        for (_, _, data) in edges:
            data['weight'] = data['weight'] / total_weight
    # Get its PageRank, alpha is 1-tau where [RAB2009 says \tau=0.15]
    page_ranks = nx.pagerank(graph, alpha=1-TAU)
    for (node, page_rank) in page_ranks.items():
        graph.node[node][PAGE_RANK] = page_rank
    return graph

def main(argv):
    "Read the supplied graph and modules and output MDL"
    # Read the arguments
    parser = argparse.ArgumentParser(description="Calculate the infomap")
    parser.add_argument('-g', '--graph-filename', type=argparse.FileType('r'),
            help="the .net file to use as the graph", required=True)
    parser.add_argument('-m', '--module-filename', default="2009_figure3a.mod",
            help="the .mod file to use as the clustering")
    options = parser.parse_args(argv[1:])

    graph = load_and_process_graph(options.graph_filename)

    # single_nodes is the "trivial" module mapping
    single_nodes = [[nodes] for nodes in graph]

    # If clustering provided, use it.
    try:
        modules = [line.strip().split() for line in options.graph_filename]
    except IOError:
        print ">>", sys.exc_info()[0]
        print ">> No .mod file provided, or error reading it"
        print ">> Using default clustering of every node in its own module"
        modules = single_nodes

    clustering = Clustering(graph, modules)
    print "This clustering has MDL %.2f (Index %.2f, Module %.2f)" % \
        (clustering.get_mdl(), clustering.get_index_codelength(),
                clustering.get_module_codelength())

if __name__ == "__main__":
    main(sys.argv)
