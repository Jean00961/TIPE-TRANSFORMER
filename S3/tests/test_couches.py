"""Tests de couches.py. Lancer depuis S3/ : python -m pytest -v tests/test_couches.py

Deux familles de tests :
1. des tests de VALEURS sur de petits exemples calculés à la main (tâches 2 et 3) ;
2. des tests de GRADIENT : le backward de chaque couche isolée est comparé aux
   différences finies (tâche 4). Ce sont eux qui garantissent que le MLP est juste.

L'astuce des tests de gradient (GUIDE.md, partie 3.9) : on fabrique une perte
artificielle f(entrée) = np.sum(C * couche.avant(entrée)), avec C une matrice
aléatoire fixe de la forme de la sortie. Pour cette perte, dL/d(sortie) = C,
donc le gradient à la main s'obtient avec couche.arriere(C).
"""
import numpy as np

# Décommentez au fur et à mesure que les classes existent :
# from couches import sigmoide, Lineaire, ReLU, Sigmoide, EntropieCroiseeBinaire
# from verif_gradient import gradient_numerique, erreur_relative


# ---------------------------------------------------------------------------
# Tâche 2 : Lineaire
# ---------------------------------------------------------------------------

def test_lineaire_formes():
    """avant et arriere renvoient les bonnes formes, dW et db ont la forme de W et b.

    Étapes :
    1. rng = np.random.default_rng(0) ; couche = Lineaire(3, 2, rng)
    2. X = rng.normal(size=(5, 3))
    3. Z = couche.avant(X) ; assert Z.shape == (5, 2)
    4. G = rng.normal(size=(5, 2)) ; dX = couche.arriere(G)
    5. assert dX.shape == (5, 3), couche.dW.shape == (3, 2), couche.db.shape == (2,)
       (une ligne assert par forme : le message d'erreur dit laquelle est fausse)
    """
    pass


def test_lineaire_valeurs_a_la_main():
    """L'exemple numérique de GUIDE.md, partie 3.3.

    Étapes :
    1. couche = Lineaire(3, 2, np.random.default_rng(0)), puis remplacez les
       poids : couche.W = np.array([[1., 2.], [3., 4.], [5., 6.]]) et
       couche.b = np.array([0.5, -0.5]).
    2. Z = couche.avant(np.ones((4, 3))) : chaque ligne vaut [9.5, 11.5].
       Astuce : np.allclose(Z, [9.5, 11.5]) compare chaque ligne grâce au
       broadcasting.
    3. dX = couche.arriere(np.ones((4, 2))) :
       - couche.dW vaut 4 partout (np.allclose(couche.dW, 4)) ;
       - couche.db vaut [4., 4.] ;
       - chaque ligne de dX vaut [3., 7., 11.].
    """
    pass


def test_init_he():
    """Les poids suivent N(0, 2 / d_entree) et les biais sont nuls.

    Étapes :
    1. couche = Lineaire(1000, 500, np.random.default_rng(0))  (500 000 poids :
       assez pour que la moyenne et l'écart-type mesurés soient précis)
    2. assert abs(couche.W.mean()) < 0.001
    3. assert np.isclose(couche.W.std(), np.sqrt(2 / 1000), rtol=0.01)
       (rtol=0.01 : on accepte 1 % d'écart relatif)
    4. assert np.all(couche.b == 0)
    """
    pass


# ---------------------------------------------------------------------------
# Tâche 3 : ReLU, Sigmoide, EntropieCroiseeBinaire
# ---------------------------------------------------------------------------

def test_relu_valeurs():
    """ReLU sur Z = [[-1, 2], [0, -3]].

    avant -> [[0, 2], [0, 0]] ; arriere(np.ones((2, 2))) -> [[0, 1], [0, 0]].
    Le 0 en position (1, 0) de arriere vient de la convention ReLU'(0) = 0.
    """
    pass


def test_sigmoide_valeurs():
    """Couche Sigmoide sur Z = np.zeros((2, 1)).

    avant -> 0.5 partout ; arriere(np.ones((2, 1))) -> 0.25 partout.
    Vérifiez aussi les formes : (2, 1) dans les deux cas.
    """
    pass


def test_entropie_valeurs():
    """Valeurs de la perte et de son gradient.

    1. perte = EntropieCroiseeBinaire()
    2. P = [[0.5], [0.5]], Y = [[0.], [1.]] :
       perte.avant(P, Y) proche de np.log(2), puis perte.arriere() proche de [[1.], [-1.]].
    3. P = [[0.9], [0.2]], Y = [[1.], [0.]] : perte proche de -(ln 0.9 + ln 0.8) / 2
       (écrivez l'expression avec np.log, pas la valeur arrondie).
    """
    pass


def test_entropie_refuse_mauvaise_forme():
    """Le piège n°1 : P de forme (2, 1) et Y de forme (2,) doit lever une erreur.

    Étapes :
    1. import pytest (en haut du fichier)
    2. with pytest.raises(AssertionError):
           EntropieCroiseeBinaire().avant(np.array([[0.5], [0.5]]), np.array([0., 1.]))
    Pour comprendre pourquoi c'est grave : dans IDLE, calculez
    np.array([[0.5], [0.5]]) - np.array([0., 1.]) et regardez la forme.
    """
    pass


def test_sigmoide_puis_perte_redonne_p_moins_y():
    """Le résultat clé de S2 retrouvé par composition : dL/dZ = (P - Y) / n.

    Étapes :
    1. rng = np.random.default_rng(0) ; Z = rng.normal(size=(6, 1)) ;
       Y = rng.integers(0, 2, size=(6, 1)).astype(float)
    2. sig = Sigmoide() ; perte = EntropieCroiseeBinaire()
    3. P = sig.avant(Z) ; perte.avant(P, Y)
    4. dZ = sig.arriere(perte.arriere())
    5. assert np.allclose(dZ, (P - Y) / 6)
    """
    pass


# ---------------------------------------------------------------------------
# Tâche 4 : tests de gradient des couches isolées
# ---------------------------------------------------------------------------
# Ordre IMPORTANT dans chaque test :
#   1. gradient à la main d'abord (avant, puis arriere), et on en garde une copie ;
#   2. gradient numérique ensuite (il appelle avant des dizaines de fois et
#      écrase le cache).
# Seuil : erreur_relative < 1e-6. Avec un backward juste, on obtient ~1e-11.

def test_gradient_lineaire_en_X():
    """dX de Lineaire contre les différences finies.

    Étapes :
    1. rng = np.random.default_rng(0) ; couche = Lineaire(3, 2, rng)
    2. couche.b = rng.normal(size=2)  (un biais non nul : plus exigeant)
    3. X = rng.normal(size=(5, 3)) ; C = rng.normal(size=(5, 2))
    4. À la main : couche.avant(X) ; dX = couche.arriere(C).copy()
    5. Numérique : g = gradient_numerique(lambda V: np.sum(C * couche.avant(V)), X)
    6. assert erreur_relative(dX, g) < 1e-6
    """
    pass


def test_gradient_lineaire_en_W():
    """dW de Lineaire contre les différences finies.

    Il faut une fonction de W seul. Les autres objets (X, C, couche) sont fixés :

        def f(W_essai):
            W_vrai = couche.W          # on garde le vrai W
            couche.W = W_essai         # on essaie le W décalé
            valeur = np.sum(C * couche.avant(X))
            couche.W = W_vrai          # on remet le vrai W : INDISPENSABLE
            return valeur

    Étapes :
    1. Même préparation que test_gradient_lineaire_en_X (rng, couche, b, X, C).
    2. À la main : couche.avant(X) ; couche.arriere(C) ; dW = couche.dW.copy()
    3. Écrivez f comme ci-dessus (à l'intérieur du test, après la préparation).
    4. g = gradient_numerique(f, couche.W)
    5. assert erreur_relative(dW, g) < 1e-6
    """
    pass


def test_gradient_lineaire_en_b():
    """db de Lineaire contre les différences finies.

    Même schéma que pour W, avec une fonction qui remplace couche.b.
    b est un vecteur (2,) : gradient_numerique le traite sans astuce.
    """
    pass


def test_gradient_relu():
    """Le backward de ReLU contre les différences finies.

    Étapes :
    1. rng = np.random.default_rng(0) ; Z = rng.normal(size=(5, 4)) ;
       C = rng.normal(size=(5, 4))
    2. relu = ReLU() ; relu.avant(Z) ; dZ = relu.arriere(C).copy()
    3. g = gradient_numerique(lambda V: np.sum(C * relu.avant(V)), Z)
    4. assert erreur_relative(dZ, g) < 1e-6

    Pourquoi c'est sûr ici : ReLU n'est pas dérivable en 0, mais le plus petit
    |Z| de ce tirage vaut environ 0.04, bien plus que eps = 1e-5. Ne testez
    jamais ReLU sur des entiers (un Z = 0 ferait échouer le test sans bug).
    """
    pass


def test_gradient_sigmoide():
    """Le backward de Sigmoide contre les différences finies.

    Même schéma que test_gradient_relu, avec Sigmoide().
    """
    pass
