import pandas as pd
import numpy as np
from igraph import *
from scipy.spatial.distance import cosine
import random


def construct_graph(attrlist, edgelist):
    """
    This function reads edges and vertex data from corresponding files and returns an attributed graph.
    """
    attributes = pd.read_csv(attrlist)
    with open(edgelist) as f:
        edges = [tuple([int(vertex) for vertex in edge.split()]) for edge in f.read().splitlines()]
    graph = Graph(n=len(attributes), edges=edges, vertex_attrs=attributes)
    graph.es['weight'] = [1] * graph.ecount()  # This is required for phase 2 while merging communities
    return graph


def construct_similarity_matrix(graph, n_vertices):
    """
    This function measures cosine similarities between vertices and returns a similarity matrix of dimension: (nV x nV)
    """
    similarities = [[None for _ in range(n_vertices)] for _ in range(n_vertices)]
    for i in range(n_vertices):
        for j in range(n_vertices):
            if not similarities[i][j]:
                # Xi and Xj are the attribute vectors (X_1, X_2, X_3, ..., X_d) for ith and jth vertices
                Xi = list(graph.vs[i].attributes().values())
                Xj = list(graph.vs[j].attributes().values())
                similarities[i][j] = similarities[j][i] = 1 - cosine(Xi, Xj)  # similarity = 1 - distance
    return similarities


def sac1(graph, alpha, n_vertices):
    """
    This function drives the SAC1 algorithm.
    """
    # Pre-compute the similarities to reduce additional operations during phase 1 and phase 2 iterations
    similarities = construct_similarity_matrix(graph, n_vertices)

    # Communities formed after phase 1
    communities_p1 = _phase1(graph, alpha, n_vertices, similarities)
    # Create a clustering from above communities
    clustering_p1 = list(VertexClustering(graph, communities_p1))

    # Meta communities formed after phase 2
    communities_p2 = _phase2(graph, alpha, communities_p1)
    # Create a clustering from above communities
    clustering_p2 = list(VertexClustering(graph, communities_p2))

    # Segregate the meta communities to obtain the final vertex memberships after successful execution of algorithm
    communities_sac1 = [[x for v in vertices for x in clustering_p1[v]] for vertices in clustering_p2]
    return communities_sac1


def _phase1(graph, alpha, n_vertices, similarities):
    """
    Phase 1 of SAC1 algorithm.
    """
    # Initialize a membership list with one vertex per community
    communities = list(range(n_vertices))

    # Dummy list to achieve random vertex selection
    vertices = list(range(n_vertices))

    run = 0  # Number of iterations
    clustered = False

    while not clustered and run < 15:
        clustered = True
        random.shuffle(vertices)  # Randomize the order of vertices to be moved
        for x in vertices:
            Q_newman = calculate_Q_newman(graph, communities, x, communities[x], move_x=False)  # Q_Newman before moving
            max_gain = 0  # Keep track of max positive ∆Q obtained so far
            max_gain_community = None  # Community corresponding to max positive ∆Q obtained so far
            for C in set(communities):  # Unique communities to reduce computation
                if communities[x] != C:
                    delta_Q_newman = calculate_Q_newman(graph, communities, x, C, move_x=True) - Q_newman  # ∆Q_Newman
                    delta_Q_attr = calculate_delta_Q_attr(communities, similarities, x, C)  # ∆Q_Attr
                    delta_Q_composite = (alpha * delta_Q_newman) + ((1 - alpha) * delta_Q_attr)  # ∆Q (composite gain)
                    if delta_Q_composite > max_gain:
                        # If composite modularity gain found during this iteration is greater than max so far...
                        max_gain = delta_Q_composite
                        max_gain_community = C
                        clustered = False
            if max_gain_community is not None:
                communities[x] = max_gain_community  # Move 'x' to community 'C' if max gain found in this iteration
        run += 1
    return flatten_communities(communities)  # Return the flattened communities for simplified igraph operations


def _phase2(graph, alpha, communities):
    """
    Phase 2 of SAC1 algorithm.
    """
    # Merge communities to form meta-vertices and merge attributes by taking mean
    graph.contract_vertices(communities, combine_attrs='mean')

    # Simplify the graph by removing duplicate edges and summing the edge weights for each community (meta-vertex now)
    graph.simplify(combine_edges='sum', loops=False)

    # Calculate new similarities for these meta-vertices and store them to reduce computation
    n_vertices = graph.vcount()
    similarities = construct_similarity_matrix(graph, n_vertices)

    # Reapply phase 1 to determine the final communities
    communities = _phase1(graph, alpha, n_vertices, similarities)
    return communities


def calculate_Q_newman(graph, communities, x, C, move_x):
    """
    Calculate the structural/Newman modularity (Q_Newman).
    """
    if not move_x:
        new_communities = communities
    else:
        new_communities = communities.copy()
        new_communities[x] = C
    return graph.modularity(new_communities, weights='weight')


def calculate_delta_Q_attr(communities, similarities, x, C):
    """
    Calculate the gain for attribute modularity (∆Q_Attr).
    """
    delta_Q_attr = 0
    comm_size = 0
    for i, comm in enumerate(communities):
        if comm == C:
            delta_Q_attr += similarities[x][i]
            comm_size += 1
    return delta_Q_attr / comm_size  # Normalize ∆Q_Attr by the size of community


def flatten_communities(communities):
    """
    Re-label the communities such that the membership of vertices remains the same but the labels for n communities
    now range from 0 to n-1. The membership attribute in igraph API requires this additional step.
    """
    comm_dict = defaultdict(set)
    for v, c in enumerate(communities):
        comm_dict[c].add(v)
    c = 0
    for vertices in comm_dict.values():
        for v in vertices:
            communities[v] = c
        c += 1
    return communities


def driver(alpha, write, fname):
    """
    A driver program that controls the execution and writes the communities formed by sac1 algorithm to a text file.
    """
    random.seed(0)  # To reproduce the results as we randomly choose a vertex 'x' to change its community membership
    graph = construct_graph(attrlist='data/fb_caltech_small_attrlist.csv',
                            edgelist='data/fb_caltech_small_edgelist.txt')
    n_vertices = graph.vcount()
    communities = sac1(graph, alpha, n_vertices)  # This is where the algorithm execution starts; returns communities
    if write:
        np.savetxt(fname=fname, X=communities, delimiter=',', fmt='%s')


if __name__ == '__main__':
    filename = {0.0: 'communities_0.txt', 0.5: 'communities_5.txt', 1.0: 'communities_1.txt'}
    write_to_file = False  # Change this to True is output needs to be written to a file
    error_msg = 'please provide one positional argument for alpha between [0, 1]'
    if len(sys.argv) != 2:
        print(error_msg)
        sys.exit(1)
    elif float(sys.argv[1]) < 0 or float(sys.argv[1]) > 1:
        print(error_msg)
        sys.exit(1)
    else:
        alpha = float(sys.argv[1])
        driver(alpha, write_to_file, filename[alpha])
