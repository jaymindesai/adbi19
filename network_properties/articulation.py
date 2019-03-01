import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from graphframes import *

sc = SparkContext("local[*]", "articulation.py")
sc.setCheckpointDir('checkpoints')
sc.setLogLevel('ERROR')

sqlContext = SQLContext(sc)


def articulations(g, usegraphframe=False):
    def _connComps(graph):
        return graph.connectedComponents().select('component').distinct().count()

    vertices = g.vertices.rdd.flatMap(list).collect()
    edges = g.edges.rdd.map(tuple).collect()
    comps = _connComps(g)
    art = []

    if usegraphframe:
        for vertex in vertices:
            new_g = GraphFrame(g.vertices.where(g.vertices['id'] != vertex),  # New vertices
                               g.edges.where(g.edges['src'] != vertex).where(g.edges['dst'] != vertex))  # New Edges
            art.append((vertex, _connComps(new_g) > comps and 1 or 0))
    else:
        for vertex in vertices:
            new_g = nx.Graph(edges)
            new_g.remove_node(vertex)  # Remove the vertex and corresponding edges
            art.append((vertex, nx.number_connected_components(new_g) > comps and 1 or 0))

    return sqlContext.createDataFrame(sc.parallelize(art), ['id', 'articulation'])


if __name__ == '__main__':
    filename = sys.argv[1]
    lines = sc.textFile(filename)

    pairs = lines.map(lambda s: s.split(","))
    e = sqlContext.createDataFrame(pairs, ['src', 'dst'])
    e = e.unionAll(e.selectExpr('src as dst', 'dst as src')).distinct()  # Ensure undirectedness

    # Extract all endpoints from input file and make a single column frame.
    v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

    # Create graphframe from the vertices and edges.
    g = GraphFrame(v, e)

    # Runtime approximately 5 minutes
    print("---------------------------")
    print("Processing graph using serial iteration over nodes and serial (NetworkX) connectedness calculations")
    init = time.time()
    df = articulations(g, False).sort('articulation', ascending=False)
    print("Execution time: %s seconds" % (time.time() - init))
    print("Articulation points:")
    df.filter('articulation = 1').show(truncate=False)
    df.toPandas().to_csv("articulations_out.csv")
    print("---------------------------")

    # Runtime for below is more than 2 hours
    # print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
    # init = time.time()
    # df = articulations(g, True)
    # print("Execution time: %s seconds" % (time.time() - init))
    # print("Articulation points:")
    # df.filter('articulation = 1').show(truncate=False)
