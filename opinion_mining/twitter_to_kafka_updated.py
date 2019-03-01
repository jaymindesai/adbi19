from json import dumps
from kafka import KafkaProducer


class TweeterStreamProducer:

    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'], batch_size=1000, linger_ms=10,
                                      value_serializer=lambda x: dumps(x).encode('utf-8'))

    def send(self, data):
        try:
            self.producer.send('twitterstream', value=data)
            self.producer.flush()
        except Exception as e:
            print(e)
            return False
        return True


if __name__ == '__main__':
    # To simulate twitter stream, we will load tweets from a file in a streaming fashion
    f = open('16M.txt')
    stream = TweeterStreamProducer()
    i = 0
    for data in f:
        stream.send(data.strip())
        i += 1
        if i % 10000 == 0:
            print("Pushed ", i, " messages")
