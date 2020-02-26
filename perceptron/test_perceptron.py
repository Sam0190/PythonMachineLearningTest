import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron.Perceptron import Perceptron
from plot_decision_regions import plot_decision_regions


def test_perceptron():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig("mygraph.png")

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal lenght [cm]')
    plt.ylabel('petal lenght [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    test_perceptron()
