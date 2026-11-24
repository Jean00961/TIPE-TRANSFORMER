"""Tests de regression_logistique.py. Lancer depuis S2/ : python -m pytest -v

Chaque test suit le même schéma :
1. préparer de petites données ;
2. appeler la fonction ;
3. assert ce qui doit être vrai.
Pour comparer des flottants, utilisez np.isclose (un nombre) ou np.allclose
(des tableaux), jamais == : 0.1 + 0.2 == 0.3 est faux en Python.
"""
import warnings

import numpy as np

# Décommentez au fur et à mesure que les fonctions existent :
# from regression_logistique import sigmoide, derivee_sigmoide, probabilites, cout, gradient, entrainer


def test_sigmoide_zero():
    """sigmoide(0) vaut 0.5.

    Une seule ligne : assert np.isclose(..., 0.5).
    """
    pass


def test_sigmoide_symetrie():
    """sigma(-z) = 1 - sigma(z) pour plusieurs valeurs de z.

    Étapes : z = np.linspace(-10, 10, 21), puis np.allclose entre les deux côtés.
    """
    pass


def test_sigmoide_stable():
    """Pas de nan ni de warning sur des valeurs extrêmes.

    Étapes :
    1. z = np.array([-1000., 0., 1000.])
    2. Appelez sigmoide dans ce bloc, qui transforme tout warning en erreur :
           with warnings.catch_warnings():
               warnings.simplefilter("error")
               s = sigmoide(z)
    3. Vérifiez qu'il n'y a aucun nan (np.isnan(...).any() doit être faux)
       et que s vaut environ [0, 0.5, 1].
    """
    pass


def test_derivee_sigmoide():
    """derivee_sigmoide est cohérente avec une différence finie sur sigmoide.

    Étapes :
    1. z = np.linspace(-5, 5, 11) et eps = 1e-5.
    2. approx = (sigmoide(z + eps) - sigmoide(z - eps)) / (2 * eps)
    3. np.allclose entre approx et derivee_sigmoide(z).
    4. Bonus : derivee_sigmoide(0) vaut 0.25.
    """
    pass


def test_probabilites_forme():
    """probabilites renvoie un tableau de forme (n,) à valeurs dans ]0, 1[.

    Étapes : X aléatoire (50, 3) avec np.random.default_rng(0), w de taille 3,
    b quelconque ; vérifiez p.shape == (50,), puis que tout p est > 0 et < 1.
    """
    pass


def test_cout_vaut_ln2_en_zero():
    """Avec w = 0 et b = 0, le coût vaut ln(2), quelles que soient les données.

    Étapes : X aléatoire (20, 4), y aléatoire de 0 et de 1 (rng.integers(0, 2, size=20)),
    w = np.zeros(4), b = 0 ; comparez à np.log(2) avec np.isclose.
    """
    pass


def test_gradient_valeur_connue():
    """gradient redonne la valeur calculée à la main dans le docstring.

    X = [[1, 2], [3, 4]], y = [0, 1], w = 0, b = 0
    -> grad_w = [-0.5, -0.5] et grad_b = 0.
    Vérifiez aussi grad_w.shape == (2,).
    """
    pass


def test_entrainement_fait_baisser_le_cout():
    """Sur des données séparables, le coût final est bien plus petit que ln(2).

    Étapes :
    1. Données : X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
       y = np.array([0., 0., 0., 1.]) (la porte ET de S1 !).
    2. w, b, couts = entrainer(X, y, eta=1.0, n_iterations=1000)
    3. Vérifiez : len(couts) == 1000, couts[0] proche de ln(2), couts[-1] < 0.1.
    4. Vérifiez que les prédictions (p > 0.5) redonnent y.
    """
    pass
