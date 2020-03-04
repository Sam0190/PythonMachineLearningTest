import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plot_decision_regions import plot_decision_regions


def test_scikit_perceptron():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    print('Accurancy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    test_scikit_perceptron()
