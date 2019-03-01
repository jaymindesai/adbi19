from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

import matplotlib.pyplot as plt
import re


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    ssc = StreamingContext(sc, 10)  # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_words("positive.txt")
    nwords = load_words("negative.txt")

    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each time-step.
    """
    pcounts, ncounts = list(), list()
    for each in counts:
        for count in each:
            if count[0] == 'positive': pcounts.append(count[1])
            else: ncounts.append(count[1])

    plt.axis([-1, len(pcounts), 0, max(*pcounts, *ncounts)+1000])
    px, nx = plt.plot(pcounts, 'ro-', ncounts, 'bo-')
    plt.xticks(range(len(pcounts)))
    plt.legend((px, nx), ('Positive', 'Negative'))
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.show()


def load_words(filename):
    """ 
    This function returns a set of words from the given filename.
    """
    words = set()
    with open(filename) as file:
        for word in file:
            words.add(word.strip())
    return words


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics=['twitterstream'], kafkaParams={"metadata.broker.list": 'localhost:9092'})

    tweets = kstream.map(lambda x: x[1])
    words = tweets.flatMap(lambda x: x.split()) \
        .map(lambda x: re.sub(r'[^\w]', '', x).lower()) \
        .filter(lambda x: x in pwords or x in nwords)

    counts = words.map(lambda x: ('positive', 1) if x in pwords else ('negative', 1)).reduceByKey(lambda x, y: x + y)

    def _updatefunc(new_values, rcount): return sum(new_values, rcount or 0)

    counts.updateStateByKey(_updatefunc).pprint()  # Running Counts

    agg_counts = []
    counts.foreachRDD(lambda x: agg_counts.append(x.collect()))

    ssc.start()  # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return agg_counts


if __name__ == "__main__":
    main()
