"""Brique 3 : la régression logistique, entraînée par descente de gradient.

Un neurone dont l'activation est la sigmoïde : p = sigma(X @ w + b).
Formes : X (n, d), y (n,), w (d,), b scalaire, p (n,).
Voir GUIDE.md, partie 1, pour la théorie et la dérivation de dL/dz = p - y.
"""
import numpy as np


def sigmoide(z):
    """Sigmoïde sigma(z) = 1 / (1 + exp(-z)), stable numériquement.

    Reçoit : z, un nombre ou un tableau NumPy de n'importe quelle forme.
    Renvoie : un tableau de même forme que z, à valeurs dans [0, 1].

    Étapes :
    1. Écrivez d'abord la version naïve, directement avec la formule.
    2. Dans IDLE, essayez sigmoide(-1000). Que voyez-vous ? (un RuntimeWarning :
       lisez-le, quel calcul déborde ?)
    3. Au tableau : montrez que 1 / (1 + exp(-z)) = exp(z) / (1 + exp(z)).
       Pour z < 0, laquelle des deux formules ne calcule jamais l'exponentielle
       d'un grand nombre positif ? Et pour z >= 0 ?
    4. Utilisez la bonne formule selon le signe de z. Attention : np.where
       calcule les DEUX branches sur tout le tableau avant de choisir, donc le
       débordement a quand même lieu. Deux pistes : travailler avec des masques
       (z >= 0 et z < 0) sur une copie, ou trouver une expression qui n'utilise
       que exp(-|z|).
    5. Convertissez z en tableau au début (np.asarray(z, dtype=float)) pour que
       la fonction marche aussi sur un simple nombre.

    Valeurs de contrôle (IDLE) :
    - sigmoide(0) -> 0.5
    - sigmoide(2) -> 0.8807970...
    - sigmoide(-2) -> 0.1192029... (= 1 - sigmoide(2))
    - sigmoide(np.array([-1000, 0, 1000])) -> [0, 0.5, 1], SANS aucun warning

    Piège : sigmoide(1000) avec la formule exp(z) / (1 + exp(z)) donne nan
    (inf / inf). C'est pour ça qu'on choisit la formule selon le signe.
    """
    pass


def derivee_sigmoide(z):
    """Dérivée de la sigmoïde : sigma'(z) = sigma(z) * (1 - sigma(z)).

    Reçoit : z, un nombre ou un tableau.
    Renvoie : un tableau de même forme que z.

    Étapes :
    1. Calculez s = sigmoide(z) une seule fois (ne l'appelez pas deux fois).
    2. Renvoyez la formule ci-dessus.

    Valeurs de contrôle :
    - derivee_sigmoide(0) -> 0.25 (c'est le maximum : pourquoi ?)
    - derivee_sigmoide(10) -> environ 4.5e-05 (presque 0 : la sigmoïde est plate)

    Piège : cette fonction ne sert pas directement dans gradient(), grâce à la
    simplification dL/dz = p - y. Elle sert à le comprendre et à le vérifier.
    """
    pass


def probabilites(X, w, b):
    """Passe avant (forward) : renvoie p = sigma(X @ w + b).

    Reçoit : X (n, d), w (d,), b scalaire.
    Renvoie : p (n,), la probabilité que chaque exemple soit de classe 1.

    Étapes :
    1. Calculez z = X @ w + b. Écrivez la forme de z en commentaire.
    2. Appliquez sigmoide.

    Valeur de contrôle :
    - X = np.array([[1., 2.], [3., 4.]]), w = np.zeros(2), b = 0
      -> [0.5, 0.5] (avec w = 0, le modèle n'a aucune préférence)

    C'est exactement neurone_batch de S1, avec sigmoide à la place de heaviside.
    """
    pass


def cout(X, y, w, b):
    """Entropie croisée binaire moyenne sur les n exemples (un nombre).

    L = - moyenne( y * ln(p) + (1 - y) * ln(1 - p) )

    Reçoit : X (n, d), y (n,), w (d,), b scalaire.
    Renvoie : un float.

    Étapes :
    1. Calculez p avec probabilites.
    2. Pour éviter ln(0) = -inf, bornez p dans [1e-12, 1 - 1e-12] (np.clip).
    3. Calculez le terme de chaque exemple (forme (n,)), puis la moyenne.
    4. N'oubliez pas le signe moins devant.

    Valeurs de contrôle :
    - Avec w = 0 et b = 0, p vaut 0.5 partout, et le coût vaut ln(2) = 0.693147...
      quelles que soient les données. (Vérifiez-le à la main : pourquoi ?)
    - X = np.array([[1., 2.], [3., 4.]]), y = np.array([0., 1.]),
      w = np.array([1., -1.]), b = 0 -> environ 0.8133
      (z = [-1, -1] : calculez p à la main, puis les deux termes)

    Pièges :
    - Le coût doit être POSITIF. S'il est négatif, il manque le signe moins.
    - Utilisez np.log (ln), pas np.log10.
    """
    pass


def gradient(X, y, w, b):
    """Gradient du coût par rapport à w et à b, calculé à la main.

    Reçoit : X (n, d), y (n,), w (d,), b scalaire.
    Renvoie : le couple (grad_w, grad_b), grad_w de forme (d,), grad_b un float.

    Formules (dérivées en GUIDE.md, partie 1.3) :
    - grad_w = X.T @ (p - y) / n
    - grad_b = moyenne de (p - y)

    Étapes :
    1. Calculez p, puis le vecteur erreur = p - y. Forme ?
    2. Récupérez n avec X.shape.
    3. Calculez grad_w. Vérifiez sa forme avec un print : elle doit être (d,),
       comme w. Si vous obtenez (n,), vous avez oublié la transposée.
    4. Calculez grad_b.

    Valeur de contrôle (à vérifier aussi à la main sur papier) :
    - X = np.array([[1., 2.], [3., 4.]]), y = np.array([0., 1.]),
      w = np.zeros(2), b = 0
      -> p = [0.5, 0.5], p - y = [0.5, -0.5],
         grad_w = [-0.5, -0.5], grad_b = 0.0

    Piège : ne divisez pas par n deux fois (une fois ici, une fois ailleurs).
    Le vrai juge sera gradient_numerique (verif_gradient.py).
    """
    pass


def entrainer(X, y, eta, n_iterations):
    """Descente de gradient : part de w = 0, b = 0 et fait n_iterations pas.

    Reçoit : X (n, d), y (n,), eta (le pas), n_iterations (un entier).
    Renvoie : w (d,), b (float), couts (liste de n_iterations floats : le coût
    AVANT chaque mise à jour).

    Étapes :
    1. Initialisez w (d zéros), b = 0.0 et une liste couts vide.
    2. À chaque itération :
       a. ajoutez cout(X, y, w, b) à la liste ;
       b. calculez (grad_w, grad_b) avec gradient ;
       c. mettez à jour w et b. Quel signe ? On DESCEND la pente.
    3. Renvoyez w, b, couts.

    Valeurs de contrôle :
    - couts[0] vaut ln(2) = 0.6931... (puisque w = 0 au départ).
    - Sur des données séparables avec eta = 0.1, la liste couts décroît.

    Pièges :
    - Avec un + au lieu d'un -, le coût augmente : c'est le premier réflexe à
      vérifier si la courbe monte.
    - Contrairement au perceptron, on ne s'arrête pas quand tout est bien
      classé : on fait toujours n_iterations pas.
    """
    pass
