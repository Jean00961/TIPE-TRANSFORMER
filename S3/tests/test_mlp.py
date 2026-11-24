"""Tests de mlp.py. Lancer depuis S3/ : python -m pytest -v tests/test_mlp.py

Dans l'ordre : la structure, les poids à la main (on SAIT que le réseau peut
résoudre XOR), le gradient du réseau entier, puis l'entraînement.
"""
import numpy as np

# Décommentez quand les classes existent :
# from couches import Lineaire, EntropieCroiseeBinaire
# from mlp import MLP, entrainer
# from verif_gradient import gradient_numerique, erreur_relative


def donnees_xor():
    """Les 4 points de XOR. Fourni : X (4, 2) et Y (4, 1) (attention, PAS (4,))."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = np.array([[0.], [1.], [1.], [0.]])
    return X, Y


def test_structure():
    """MLP([2, 4, 1]) a 4 couches dans le bon ordre et 17 paramètres.

    Étapes :
    1. mlp = MLP([2, 4, 1])
    2. noms = [type(c).__name__ for c in mlp.couches]
       assert noms == ['Lineaire', 'ReLU', 'Lineaire', 'Sigmoide']
    3. assert mlp.nombre_parametres() == 17
    4. Même chose pour MLP([2, 32, 32, 1]) : 6 couches, 1185 paramètres.
    """
    pass


def test_formes_avant_arriere():
    """avant renvoie (n, 1) ; après arriere, chaque dW a la forme de son W.

    Étapes :
    1. mlp = MLP([2, 4, 1]) ; X, Y = donnees_xor()
    2. P = mlp.avant(X) ; assert P.shape == (4, 1)
    3. mlp.arriere(np.ones((4, 1)))
    4. Pour chaque couche c qui est une Lineaire :
       assert c.dW.shape == c.W.shape et c.db.shape == c.b.shape
    """
    pass


def test_valeurs_initiales():
    """Avec la graine 0, on retrouve exactement les valeurs de la référence.

    Étapes :
    1. mlp = MLP([2, 4, 1], graine=0) ; X, Y = donnees_xor()
    2. np.allclose(mlp.couches[0].W[0], [0.1257, -0.1321, 0.6404, 0.1049], atol=1e-4)
    3. P = mlp.avant(X) ; np.allclose(P.ravel(), [0.5, 0.2951, 0.4154, 0.2628], atol=1e-4)
    4. np.isclose(EntropieCroiseeBinaire().avant(P, Y), 0.7742, atol=1e-4)

    Si ce test échoue mais que tous les autres passent, regardez l'appel à
    rng.normal dans Lineaire.__init__ (docstring) : ce n'est pas grave pour
    l'apprentissage, mais on perd la reproductibilité avec la référence.
    """
    pass


def test_poids_a_la_main_resolvent_xor():
    """Les poids de GUIDE.md, partie 3.1, résolvent XOR sans aucun entraînement.

    Étapes :
    1. mlp = MLP([2, 2, 1])
    2. Remplacez les poids :
       couche 0 : W = [[1., 1.], [1., 1.]], b = [0., -1.]
       couche 2 : W = [[10.], [-20.]], b = [-5.]
    3. X, Y = donnees_xor() ; assert np.array_equal(mlp.predire(X), Y.astype(int))
    4. Bonus : np.allclose(mlp.avant(X).ravel(), [0.0067, 0.9933, 0.9933, 0.0067], atol=1e-4)

    Intérêt : on sait maintenant qu'une solution EXISTE avec 2 neurones cachés.
    Si l'entraînement échoue, ce n'est pas l'architecture qui est en cause.
    """
    pass


def test_gradient_mlp_entier():
    """Le test le plus important : le backward du réseau ENTIER contre les différences finies.

    Si chaque couche est juste mais que celui-ci échoue, l'erreur est dans
    l'assemblage (MLP.arriere).

    Étapes :
    1. mlp = MLP([2, 4, 1], graine=1) ; X, Y = donnees_xor() ;
       perte = EntropieCroiseeBinaire()
    2. À la main : perte.avant(mlp.avant(X), Y) ; mlp.arriere(perte.arriere())
       couche = mlp.couches[0] ; dW = couche.dW.copy()
    3. Fonction du W de la PREMIÈRE couche (le gradient a traversé tout le réseau) :
           def f(W_essai):
               W_vrai = couche.W
               couche.W = W_essai
               valeur = perte.avant(mlp.avant(X), Y)
               couche.W = W_vrai
               return valeur
    4. assert erreur_relative(dW, gradient_numerique(f, couche.W)) < 1e-6
       (référence : environ 1e-10)
    5. Faites de même pour le W de la dernière couche linéaire (mlp.couches[2]).

    Pourquoi la graine 1 et pas 0 ? Pour ne pas tester toujours le même point.
    """
    pass


def test_entrainement_resout_xor():
    """Après entraînement, le 2-4-1 classe correctement les 4 points de XOR.

    Étapes :
    1. mlp = MLP([2, 4, 1], graine=0) ; X, Y = donnees_xor()
    2. couts = entrainer(mlp, X, Y, eta=0.5, n_iterations=2000)
    3. assert len(couts) == 2000
    4. assert couts[-1] < 0.01   (référence : environ 0.0020)
    5. assert np.array_equal(mlp.predire(X), Y.astype(int))
    """
    pass


def test_init_constante_garde_les_neurones_identiques():
    """BONUS B1 : avec des poids tous égaux, les neurones cachés restent des clones.

    Étapes :
    1. mlp = MLP([2, 4, 1], graine=0) ; X, Y = donnees_xor()
    2. Pour chaque couche linéaire c : c.W[:] = 0.5  (le [:] remplit le tableau
       existant au lieu d'en créer un nouveau)
    3. entrainer(mlp, X, Y, eta=0.5, n_iterations=2000)
    4. W1 = mlp.couches[0].W : toutes ses colonnes sont égales à la première.
       assert np.allclose(W1, W1[:, [0]])   (W1[:, [0]] a la forme (2, 1) :
       le broadcasting la compare à chaque colonne)
    5. XOR n'est PAS résolu : assert not np.array_equal(mlp.predire(X), Y.astype(int))

    C'est la preuve expérimentale de l'argument de symétrie (GUIDE.md, 3.7).
    """
    pass
