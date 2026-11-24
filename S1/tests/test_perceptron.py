import numpy as np

from neurone import neurone_batch
from perceptron import entrainer_perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def test_et():
    """Le perceptron apprend ET : erreurs[-1] == 0 et neurone_batch(X, w, b) redonne exactement y."""
    y = np.array([0, 0, 0, 1])
    w, b, erreurs = entrainer_perceptron(X, y, 0.1, 100)
    assert erreurs[-1] == 0
    assert np.array_equal(neurone_batch(X, w, b), y)


def test_ou():
    """Le perceptron apprend OU : erreurs[-1] == 0 et neurone_batch(X, w, b) redonne exactement y."""
    y = np.array([0, 1, 1, 1])
    w, b, erreurs = entrainer_perceptron(X, y, 0.1, 100)
    assert erreurs[-1] == 0
    assert np.array_equal(neurone_batch(X, w, b), y)


def test_xor_echoue():
    """Sur XOR, après 100 époques, min(erreurs) > 0 : le test documente un échec attendu."""
    y = np.array([0, 1, 1, 0])
    w, b, erreurs = entrainer_perceptron(X, y, 0.1, 100)
    assert len(erreurs) == 100  # jamais d'arrêt anticipé
    assert min(erreurs) > 0
