import sys
import time
import collections
import nltk
import random
import pickle

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from numpy import zeros, empty, isnan, random, uint32, float32 as REAL, vstack
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# from gensim.models.deprecated.doc2vec import TaggedDocument
# from gensim_models import word2vec
# from gensim_models import doc2vec_modified
# from gensim_models.doc2vec_modified import TaggedDocument, Doc2Vec

random.seed(0)
INF = float("inf")

nltk.download("stopwords")  # Download the stop words from nltk

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print("python sentiment.py <path_to_data> <nlp|d2v|w2v>")
    exit(1)
path_to_data = sys.argv[1]
method = sys.argv[2]

if method == "w2v":
    path_to_pretrained_w2v = ""


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    # Using saved models and vectors for method == 'nlp'. (Original runtime = 5 minutes; Current runtime = 10 seconds)
    if method == "nlp":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)

        filename = './' + path_to_data + 'train_pos_vec_nlp.txt'
        # filename = path_to_data + 'train_pos_vec_nlp.txt'
        pickle.dump(train_pos_vec, open(filename, 'wb'))
        train_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'train_neg_vec_nlp.txt'
        # filename = path_to_data + 'train_neg_vec_nlp.txt'
        pickle.dump(train_neg_vec, open(filename, 'wb'))
        train_neg_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_pos_vec_nlp.txt'
        # filename = path_to_data + 'test_pos_vec_nlp.txt'
        pickle.dump(test_pos_vec, open(filename, 'wb'))
        test_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_neg_vec_nlp.txt'
        # filename = path_to_data + 'test_neg_vec_nlp.txt'
        pickle.dump(test_neg_vec, open(filename, 'wb'))
        test_neg_vec = pickle.load(open(filename, 'rb'))

        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)

        filename = './' + path_to_data + 'nb_model_nlp.sav'
        # filename = path_to_data + 'nb_model_nlp.sav'
        pickle.dump(nb_model, open(filename, 'wb'))
        nb_model = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'lr_model_nlp.sav'
        # filename = path_to_data + 'lr_model_nlp.sav'
        pickle.dump(lr_model, open(filename, 'wb'))
        lr_model = pickle.load(open(filename, 'rb'))

    # Using saved models and vectors for method == 'd2v'. (Original runtime = 10 minutes; Current runtime = 10 seconds)
    if method == "d2v":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos,
                                                                                    test_neg)
        filename = './' + path_to_data + 'train_pos_vec_d2v.txt'
        # filename = path_to_data + 'train_pos_vec_d2v.txt'
        pickle.dump(train_pos_vec, open(filename, 'wb'))
        train_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'train_neg_vec_d2v.txt'
        # filename = path_to_data + 'train_neg_vec_d2v.txt'
        pickle.dump(train_neg_vec, open(filename, 'wb'))
        train_neg_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_pos_vec_d2v.txt'
        # filename = path_to_data + 'test_pos_vec_d2v.txt'
        pickle.dump(test_pos_vec, open(filename, 'wb'))
        test_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_neg_vec_d2v.txt'
        # filename = path_to_data + 'test_neg_vec_d2v.txt'
        pickle.dump(test_neg_vec, open(filename, 'wb'))
        test_neg_vec = pickle.load(open(filename, 'rb'))

        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)

        filename = './' + path_to_data + 'nb_model_d2v.sav'
        # filename = path_to_data + 'nb_model_d2v.sav'
        pickle.dump(nb_model, open(filename, 'wb'))
        nb_model = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'lr_model_d2v.sav'
        # filename = path_to_data + 'lr_model_d2v.sav'
        pickle.dump(lr_model, open(filename, 'wb'))
        lr_model = pickle.load(open(filename, 'rb'))

    if method == "w2v":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC_W2V(train_pos, train_neg, test_pos,
                                                                                        test_neg)
        filename = './' + path_to_data + 'train_pos_vec_w2v.txt'
        # filename = path_to_data + 'train_pos_vec_w2v.txt'
        pickle.dump(train_pos_vec, open(filename, 'wb'))
        # train_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'train_neg_vec_w2v.txt'
        # filename = path_to_data + 'train_neg_vec_w2v.txt'
        pickle.dump(train_neg_vec, open(filename, 'wb'))
        # train_neg_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_pos_vec_w2v.txt'
        # filename = path_to_data + 'test_pos_vec_w2v.txt'
        pickle.dump(test_pos_vec, open(filename, 'wb'))
        # test_pos_vec = pickle.load(open(filename, 'rb'))

        filename = './' + path_to_data + 'test_neg_vec_w2v.txt'
        # filename = path_to_data + 'test_neg_vec_w2v.txt'
        pickle.dump(test_neg_vec, open(filename, 'wb'))
        # test_neg_vec = pickle.load(open(filename, 'rb'))

        nb_model, lr_model = build_models_DOC_W2V(train_pos_vec, train_neg_vec)

        filename = './' + path_to_data + 'nb_model_w2v.sav'
        # filename = path_to_data + 'nb_model_w2v.sav'
        pickle.dump(nb_model, open(filename, 'wb'))

        filename = './' + path_to_data + 'lr_model_w2v.sav'
        # filename = path_to_data + 'lr_model_w2v.sav'
        pickle.dump(lr_model, open(filename, 'wb'))

    print("Naive Bayes")
    print("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)

    print("")
    print("Logistic Regression")
    print("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir + "train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            if (len(words) == 0): continue
            train_pos.append(words)
    with open(path_to_dir + "train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            if (len(words) == 0): continue
            train_neg.append(words)
    with open(path_to_dir + "test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            if (len(words) == 0): continue
            test_pos.append(words)
    with open(path_to_dir + "test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            if (len(words) == 0): continue
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many positive texts as negative texts, or vice-versa.

    def _counter(tweets):
        counter = collections.defaultdict(int)
        for tweet in tweets:
            for word in set(tweet) - stopwords:  # condition_1: remove stopwords
                # Occurrence of words in tweets, for example: ('hello': 4) => 'hello' appeared in 4 tweets.
                counter[word] += 1
        return counter

    pos_tweets = len(train_pos)
    neg_tweets = len(train_neg)

    pos_word_counts = _counter(train_pos)
    neg_word_counts = _counter(train_neg)

    features = set()

    # Filter the words with condition_2 and condition_3, populate feature set.
    def _filter(to_filter, other):
        for word, occurrence in to_filter.items():
            cond_2 = occurrence >= 0.01 * pos_tweets or occurrence >= 0.01 * neg_tweets
            cond_3 = occurrence >= 2 * other.get(word, INF) or 2 * occurrence <= other.get(word, 0)
            if cond_2 and cond_3:
                features.add(word)

    _filter(pos_word_counts, neg_word_counts)
    _filter(neg_word_counts, pos_word_counts)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.

    feature_indices = {feature: idx for idx, feature in enumerate(features)}

    # Vectorize tweets into tweet by word-occurrence matrix.
    def _to_matrix(tweets):
        matrix = []
        for tweet in tweets:
            row = [0] * len(features)
            for word in set(tweet):
                if word in features:
                    row[feature_indices[word]] = 1
            matrix.append(row)
        return matrix

    train_pos_vec = _to_matrix(train_pos)
    train_neg_vec = _to_matrix(train_neg)
    test_pos_vec = _to_matrix(test_pos)
    test_neg_vec = _to_matrix(test_neg)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires TaggedDocument objects as input.
    # Turn the datasets from lists of words to lists of TaggedDocument objects.

    def _to_labeled(label, tweets):
        return [TaggedDocument(tweet, [label + str(i)]) for i, tweet in enumerate(tweets)]

    labeled_train_pos = _to_labeled("TRAIN_POS_", train_pos)
    labeled_train_neg = _to_labeled("TRAIN_NEG_", train_neg)
    labeled_test_pos = _to_labeled("TEST_POS_", test_pos)
    labeled_test_neg = _to_labeled("TEST_NEG_", test_neg)

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    print("Doc2Vec")
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    print("end of training")

    # Use the docvecs function to extract the feature vectors for the training and test data
    def _vectorize(label, tweets):
        return [model.docvecs[label + str(i)] for i in range(len(tweets))]

    train_pos_vec = _vectorize("TRAIN_POS_", train_pos)
    train_neg_vec = _vectorize("TRAIN_NEG_", train_neg)
    test_pos_vec = _vectorize("TEST_POS_", test_pos)
    test_neg_vec = _vectorize("TEST_NEG_", test_neg)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def feature_vecs_DOC_W2V(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Load the pre-trained word2vec model
    word2vec_model = word2vec.Word2Vec.load(path_to_pretrained_w2v)

    # Doc2Vec requires TaggedDocument objects as input.
    # Turn the datasets from lists of words to lists of TaggedDocument objects.
    labeled_train_pos = [TaggedDocument(words, ["TRAIN_POS_" + str(i)]) for i, words in enumerate(train_pos)]
    labeled_train_neg = [TaggedDocument(words, ["TRAIN_NEG_" + str(i)]) for i, words in enumerate(train_neg)]
    labeled_test_pos = [TaggedDocument(words, ["TEST_POS_" + str(i)]) for i, words in enumerate(test_pos)]
    labeled_test_neg = [TaggedDocument(words, ["TEST_NEG_" + str(i)]) for i, words in enumerate(test_neg)]

    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg

    # Use modified doc2vec codes for applying the pre-trained word2vec model
    model = doc2vec_modified.Doc2Vec(dm=0, dm_mean=1, alpha=0.025, min_alpha=0.0001, min_count=1, size=1000, hs=1,
                                     workers=4, train_words=False, train_lbls=True)
    model.reset_weights()

    # Copy wiki word2vec model into doc2vec model
    model.vocab = word2vec_model.vocab
    model.syn0 = word2vec_model.syn0
    model.syn1 = word2vec_model.syn1
    model.index2word = word2vec_model.index2word

    print("# of pre-trained vocab = " + str(len(model.vocab)))

    # Extract sentence labels for the training and test data
    train_pos_labels = ["TRAIN_POS_" + str(i) for i in range(len(labeled_train_pos))]
    train_neg_labels = ["TRAIN_NEG_" + str(i) for i in range(len(labeled_train_neg))]
    test_pos_labels = ["TEST_POS_" + str(i) for i in range(len(labeled_test_pos))]
    test_neg_labels = ["TEST_NEG_" + str(i) for i in range(len(labeled_test_neg))]

    sentence_labels = train_pos_labels + train_neg_labels + test_pos_labels + test_neg_labels

    new_syn0 = empty((len(sentences), model.layer1_size), dtype=REAL)
    new_syn1 = empty((len(sentences), model.layer1_size), dtype=REAL)

    syn_index = 0

    # Initialize and add a vector of syn0 (i.e. input vector) and syn1 (i.e. output vector) for a vector of a label
    for label in sentence_labels:
        v = model.append_label_into_vocab(label)  # I made this function in the doc2vec code

        random.seed(uint32(model.hashfxn(model.index2word[v.index] + str(model.seed))))

        new_syn0[syn_index] = (random.rand(model.layer1_size) - 0.5) / model.layer1_size
        new_syn1[syn_index] = zeros((1, model.layer1_size), dtype=REAL)

        syn_index += 1

    model.syn0 = vstack([model.syn0, new_syn0])
    model.syn1 = vstack([model.syn1, new_syn1])

    model.precalc_sampling()

    # Train the model
    # This may take a bit to run
    for i in range(5):
        start_time = time.time()

        print("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences)

        print("Done - Training")
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))
        start_time = time.time()

        # Convert "nan" values into "0" in vectors
        indices_nan = isnan(model.syn0)
        model.syn0[indices_nan] = 0.0

        indices_nan = isnan(model.syn1)
        model.syn1[indices_nan] = 0.0

        # Extract the feature vectors for the training and test data
        train_pos_vec = [model.syn0[model.vocab["TRAIN_POS_" + str(i)].index] for i in range(len(labeled_train_pos))]
        train_neg_vec = [model.syn0[model.vocab["TRAIN_NEG_" + str(i)].index] for i in range(len(labeled_train_neg))]
        test_pos_vec = [model.syn0[model.vocab["TEST_POS_" + str(i)].index] for i in range(len(labeled_test_pos))]
        test_neg_vec = [model.syn0[model.vocab["TEST_NEG_" + str(i)].index] for i in range(len(labeled_test_neg))]

        print("Done - Extracting the feature vectors")
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LogisticRegression Model that are fit to the training data.
    """
    X_train = train_pos_vec + train_neg_vec
    y_train = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X_train, y_train)

    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LogisticRegression Model that are fit to the training data.
    """
    X_train = train_pos_vec + train_neg_vec
    y_train = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    return nb_model, lr_model


def build_models_DOC_W2V(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    X = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X, Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.

    X_test = test_pos_vec + test_neg_vec
    y_test = ["pos"] * len(test_pos_vec) + ["neg"] * len(test_neg_vec)

    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()

    accuracy = float(tp + tn) / float(tn + tp + fn + fp)

    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d" % (tp, fn))
        print("neg\t\t%d\t%d" % (fp, tn))
    print("accuracy: %f" % (accuracy))


if __name__ == "__main__":
    main()
