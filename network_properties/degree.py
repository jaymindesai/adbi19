import sys
import networkx as nx
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from graphframes import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

sc = SparkContext("local", "degree.py")
sc.setLogLevel('ERROR')

sqlContext = SQLContext(sc)


def _df(x, schema=None): return sqlContext.createDataFrame(x, schema)


def simple(g):
    """
        Return the simple closure of the graph as a graphframe.
    """
    edges = g.edges.rdd.map(tuple)
    flipped = edges.map(lambda x: tuple(reversed(x)))
    edges = edges.union(flipped).distinct().filter(lambda x: x[0] != x[1])

    return GraphFrame(g.vertices, _df(edges, ['src', 'dst']))


def degreeDist(g):
    """
        Return a data frame of the degree distributions of each edge in
	    the provided graphframe.
    """
    return g.inDegrees.selectExpr('inDegree as degree').groupBy('degree').count()


def readFile(filename, large):
    """
        Read in an edgelist file with lines of the format id1<delim>id2
	    and return a corresponding graphframe. If "large" we assume
	    a header row and that delim = " ", otherwise no header and
	    delim = ","
    """
    lines = sc.textFile(filename)

    if large:
        delim = " "
        # Strip off header row.
        lines = lines.mapPartitionsWithIndex(lambda ind, it: iter(list(it)[1:]) if ind == 0 else it)
    else:
        delim = ","

    edges = lines.map(lambda l: l.split(delim)).map(lambda x: Row(src=x[0], dst=x[1]))
    vertices = lines.flatMap(lambda l: l.split(delim)).distinct().map(lambda x: Row(id=x))

    return GraphFrame(_df(vertices), _df(edges))


def scaleFree(dist, nv, name, plot=False):
    """
        Check if the graph is scale-free.
        :param dist: Degree distribution of the graph
        :param nv: Number of vertices in graph
        :param name: Name of graph
        :param plot: True/False whether to plot the graph of Fraction P(k) vs. Degree 'k'
        :return: True if graph is scale-free, False otherwise
    """
    if plot: dist = dist.sort('degree')

    k = dist.select('degree').rdd.map(lambda x: x['degree']).collect()
    pk = dist.select('count').rdd.map(lambda x: x['count'] / nv).collect()

    # Fit the line y = mx without intercept where y = log(P(k)), x = log(k) and m = -gamma
    gamma = -sm.OLS(np.log(pk), np.log(k)).fit().params[0]
    print('Gamma for {}: {}'.format(name, gamma))

    if plot: _plotDD(k, pk, name)

    # We check whether gamma lies between 2 and 3 to check for existence of power law property in the distribution
    return 2 < gamma < 3 and True or False


def _plotDD(k, pk, name):
    """
        Plot the degree distribution of the nodes.
        :param k: An array with all possible degrees in graph
        :param pk: An array containing fraction of nodes with degree 'k'
        :param name: Filename for the plot
    """
    plt.plot(k, pk, 'ro-')
    plt.xlabel('Degree (k)')
    plt.ylabel('Fraction P(k)')
    plt.savefig(name + '.png')


if __name__ == '__main__':
    WRITE = False
    PLOT = False

    # Generate graph from file specified
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == 'large':
            large = True
        else:
            large = False

        print("Processing input file " + filename)
        g = readFile(filename, large)

        print("Original graph has {} directed edges and {} vertices.".format(g.edges.count(), g.vertices.count()))

        g2 = simple(g)
        print("Simple graph has " + str(g2.edges.count() // 2) + " undirected edges.")

        distrib = degreeDist(g2)
        # distrib.show()
        nodecount = g2.vertices.count()
        print("Graph has " + str(nodecount) + " vertices.")

        out = filename.split("/")[-1]
        sf = scaleFree(distrib, nodecount, out, PLOT) and 'scale-free' or 'not scale-free'
        print('Graph {} is {}.'.format(out, sf))

        if WRITE:
            print("Writing distribution to file " + out + ".csv")
            distrib.toPandas().to_csv(out + ".csv")

    # Generate some random graphs if file not specified
    else:
        print("Generating random graphs.")
        vschema = Row('id')
        eschema = Row('src', 'dst')

        gnp1 = nx.gnp_random_graph(100, 0.05, seed=1234)
        gnp2 = nx.gnp_random_graph(2000, 0.01, seed=5130303)
        gnm1 = nx.gnm_random_graph(100, 1000, seed=27695)
        gnm2 = nx.gnm_random_graph(1000, 100000, seed=9999)
        gscf1 = nx.scale_free_graph(1000)
        gscf2 = nx.scale_free_graph(5000)

        todo = {"gnp1": gnp1, "gnp2": gnp2, "gnm1": gnm1, "gnm2": gnm2, "gscf1": gscf1, "gscf2": gscf2}
        for gx in todo:
            print("Processing graph " + gx)
            v = _df(sc.parallelize(todo[gx].nodes()).map(vschema))
            e = _df(sc.parallelize(todo[gx].edges()).map(lambda x: eschema(*x)))
            g = simple(GraphFrame(v, e))

            distrib = degreeDist(g)
            nodecount = g.vertices.count()

            sf = scaleFree(distrib, nodecount, gx, PLOT) and 'scale-free' or 'not scale-free'
            print('Graph {} is {}.'.format(gx, sf))

            if WRITE:
                print("Writing distribution to file " + gx + ".csv")
                distrib.toPandas().to_csv(gx + ".csv")
