import numpy as np

from neurone import heaviside, neurone, neurone_batch


def test_heaviside():
    """Vérifier heaviside sur un nombre et sur un tableau (valeurs négatives, nulle, positives)."""
    assert heaviside(-2.5) == 0
    assert heaviside(0) == 1  # convention : z >= 0 donne 1
    assert heaviside(3) == 1
    z = np.array([-1.0, 0.0, 0.5, 10.0])
    assert np.array_equal(heaviside(z), np.array([0, 1, 1, 1]))


def test_batch_egal_boucle():
    """Sur un X aléatoire de forme (50, 3) (np.random.default_rng(0)),
    neurone_batch et une boucle sur neurone donnent le même résultat
    (np.array_equal), et la sortie a la forme (50,)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w = rng.normal(size=3)
    b = 0.3
    sortie_batch = neurone_batch(X, w, b)
    sortie_boucle = np.array([neurone(x, w, b) for x in X])
    assert sortie_batch.shape == (50,)
    assert np.array_equal(sortie_batch, sortie_boucle)
