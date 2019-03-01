from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as fn
from graphframes import *

sc = SparkContext("local[*]", "centrality.py")
sc.setLogLevel('ERROR')

sqlContext = SQLContext(sc)


def closeness(g):
    return g.shortestPaths(g.vertices.rdd.flatMap(list).collect()) \
        .select(fn.explode('distances')) \
        .groupBy('key').agg(fn.sum('value').alias('sum')) \
        .selectExpr('key as id', '1/sum as closeness')


if __name__ == '__main__':
    print("Reading in graph for problem 2.")
    graph = sc.parallelize([('A', 'B'), ('A', 'C'), ('A', 'D'),
                            ('B', 'A'), ('B', 'C'), ('B', 'D'), ('B', 'E'),
                            ('C', 'A'), ('C', 'B'), ('C', 'D'), ('C', 'F'), ('C', 'H'),
                            ('D', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'E'), ('D', 'F'), ('D', 'G'),
                            ('E', 'B'), ('E', 'D'), ('E', 'F'), ('E', 'G'),
                            ('F', 'C'), ('F', 'D'), ('F', 'E'), ('F', 'G'), ('F', 'H'),
                            ('G', 'D'), ('G', 'E'), ('G', 'F'),
                            ('H', 'C'), ('H', 'F'), ('H', 'I'),
                            ('I', 'H'), ('I', 'J'),
                            ('J', 'I')])

    e = sqlContext.createDataFrame(graph, ['src', 'dst'])
    v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
    print("Generating GraphFrame.")
    g = GraphFrame(v, e)

    print("Calculating Closeness.")
    closeness(g).sort('closeness', ascending=False).toPandas().to_csv("centrality_out.csv")
