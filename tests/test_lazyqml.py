#!/usr/bin/env python

"""Tests for `lazyqml` package."""


import unittest

from lazyqml.supervised import QuantumClassifier


class TestLazyqml(unittest.TestCase):
    """Tests for `lazyqml` package."""

    def test_multiclass(self):
        from sklearn.datasets import load_breast_cancer, load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X = data.data
        y = data.target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

        q = QuantumClassifier(nqubits=4,classifiers=["all"])

        scores = q.fit(X_train, X_test, y_train, y_test)

        print(scores)

    def test_binary(self):
        from sklearn.datasets import load_breast_cancer, load_iris
        from sklearn.model_selection import train_test_split
        data = load_breast_cancer()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

        q = QuantumClassifier(nqubits=2,classifiers=["all"])

        scores = q.fit(X_train, X_test, y_train, y_test)

        print(scores)