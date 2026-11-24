# S2 : Apprendre par descente de gradient

Ce guide vous accompagne pas à pas. Il ne donne pas le code : il donne l'ordre, les questions à se poser, les dimensions et des **valeurs de contrôle** à vérifier dans IDLE. Si votre fonction redonne la valeur de contrôle, vous êtes sur la bonne voie.

Règle d'or de la séance : **une fonction, puis son test, puis la suivante.** Ne passez jamais à la fonction suivante tant que la précédente ne donne pas la valeur de contrôle.

---

## 0. Avant tout : finir S1 (≈ 30 min)

S1 n'est pas terminé : `neurone.py` est fini, mais `perceptron.py`, les tests et la figure sont vides, et `journal.md` aussi. S2 ne réutilise pas ce code, mais le plan demande de ne jamais commencer une brique tant que la précédente n'a pas ses tests.

### 0.1 `S1/src/perceptron.py` : `entrainer_perceptron`

Écrivez d'abord sur papier ce que fait **une** époque, puis codez.

1. Combien d'exemples `n` et de coordonnées `d` y a-t-il ? (`X.shape`)
2. Initialisez `w` à un vecteur de `d` zéros, `b` à 0, et une liste `erreurs` vide.
3. Boucle extérieure sur les époques. Dans chaque époque, un compteur d'erreurs part de 0.
4. Boucle intérieure sur les exemples `i`. Pour **un** exemple :
   - calculez la prédiction `y_chapeau` avec la fonction `neurone` de S1 (un seul exemple : `X[i]`) ;
   - calculez l'écart `y[i] - y_chapeau`. Quelles valeurs peut-il prendre ? (trois possibilités)
   - si l'écart n'est pas nul : appliquez la règle `w ← w + η·écart·x` **et** `b ← b + η·écart`, et ajoutez 1 au compteur.
5. À la fin de l'époque, ajoutez le compteur à `erreurs`. S'il vaut 0, sortez de la boucle (`break`).
6. Renvoyez `w, b, erreurs`.

Pour importer `neurone` dans `perceptron.py`, écrivez `from neurone import neurone` en haut du fichier : `pytest.ini` rend `src/` visible.

**Valeur de contrôle (IDLE, depuis `S1/src`)** : sur ET avec `eta=1`, `n_epoques=100`, la liste `erreurs` doit finir par `0`, et `neurone_batch(X, w, b)` doit redonner `[0, 0, 0, 1]`. Sur XOR, `erreurs` ne contient jamais 0.

**Pièges :**
- Oublier de mettre à jour `b` : ET ne converge alors pas (pourquoi ? que vaut la sortie pour `x = (0, 0)` si `b` reste à 0 ?).
- Remettre le compteur à 0 au mauvais endroit (dans la boucle sur les exemples au lieu de la boucle sur les époques).

### 0.2 Les tests de S1

Remplacez chaque `pass` par du vrai code. Un test, c'est : préparer des données, appeler la fonction, puis `assert` ce qui doit être vrai.
- `test_heaviside` : `assert heaviside(0) == 1`, puis pour un nombre négatif, puis pour un tableau avec `np.array_equal`.
- `test_batch_egal_boucle` : `rng = np.random.default_rng(0)`, puis `X = rng.normal(size=(50, 3))`, un `w` de taille 3 et un `b`. Construisez la version boucle avec une liste de `neurone(x, w, b)` pour chaque ligne `x` de `X`, puis comparez.
- `test_et`, `test_ou`, `test_xor_echoue` : suivez les docstrings. Pensez à importer les fonctions en haut du fichier.

**Vérification** : dans `S1/`, `python -m pytest -v` doit afficher 5 tests `PASSED`. Faites-en échouer un exprès (changez `y_et`) pour vérifier que le test sert à quelque chose.

### 0.3 Figure et journal

La figure `s1_frontieres.png` peut attendre (marquez-la en reliquat dans le journal). Remplissez en revanche `S1/journal.md` en 5 lignes : *Fait*, *Compris*, *Bloqué*, *Étonnement*, *Première action de la prochaine séance*.

---

## 1. Théorie (25 min, au tableau)

### 1.1 Du perceptron à la régression logistique

Le perceptron renvoie 0 ou 1 : on ne peut pas dériver un échelon. On remplace l'échelon par la **sigmoïde**, qui donne une probabilité :

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = w \cdot x + b$$

À faire au tableau :
1. Tracez l'allure de σ. Que vaut σ(0) ? Les limites en ±∞ ?
2. Montrez que σ(−z) = 1 − σ(z).
3. Montrez que **σ'(z) = σ(z)(1 − σ(z))**. Indice : dérivez $(1 + e^{-z})^{-1}$, puis écrivez $\frac{e^{-z}}{1 + e^{-z}}$ comme $1 - \sigma(z)$.

### 1.2 Le coût : entropie croisée binaire

Pour un exemple : $L = -[\,y \ln p + (1 - y)\ln(1 - p)\,]$.

Questions à se poser :
- Si `y = 1`, quel terme reste ? Que vaut L quand p → 1 ? quand p → 0 ? Pourquoi est-ce le comportement voulu ?
- Sur un batch de n exemples, on prend la **moyenne** des L.

### 1.3 Le résultat clé : ∂L/∂z = p − y (10 min)

Dérivation guidée, pour un seul exemple :
1. Calculez $\frac{\partial L}{\partial p}$. (Deux termes, chacun avec une fraction.)
2. Écrivez la règle de la chaîne : $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p}\cdot\frac{\partial p}{\partial z}$.
3. Remplacez $\frac{\partial p}{\partial z}$ par $p(1-p)$ et simplifiez. Tout doit se réduire à **p − y**.

Ensuite, comme $z = w \cdot x + b$ : $\frac{\partial z}{\partial w_j} = x_j$ et $\frac{\partial z}{\partial b} = 1$. D'où, sur un batch :

$$\nabla_w = \frac{1}{n} X^\top (p - y), \qquad \frac{\partial L}{\partial b} = \text{moyenne}(p - y)$$

### 1.4 Tableau des dimensions (à recopier sur la fiche n°2)

| Objet | Forme | Remarque |
| --- | --- | --- |
| `X` | `(n, d)` | une ligne par exemple |
| `y` | `(n,)` | des 0 et des 1 |
| `w` | `(d,)` | |
| `b` | scalaire | |
| `z = X @ w + b` | `(n,)` | broadcasting de `b` |
| `p` | `(n,)` | |
| `L` | scalaire | moyenne sur n |
| `X.T` | `(d, n)` | |
| `X.T @ (p - y)` | `(d,)` | même forme que `w` : normal, c'est son gradient |

Vérifiez la dernière ligne : `(d, n) @ (n,)` donne bien `(d,)`.

### 1.5 Descente de gradient et rôle de η

À chaque itération : `w ← w − η ∇w` et `b ← b − η ∂b`. Attention au **signe moins** : on descend la pente.
- η trop petit : le coût baisse, mais très lentement.
- η trop grand : le coût oscille, voire augmente.
- Différence avec S1 : ici η compte vraiment (relisez l'étonnement du bonus de S1).

### 1.6 Vérifier un gradient : différences finies centrées

On ne fait jamais confiance à un gradient calculé à la main. On le compare à :

$$\frac{\partial f}{\partial w_i} \approx \frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}$$

où $e_i$ est le vecteur qui vaut 1 en position i et 0 ailleurs. On compare avec l'**erreur relative** :

$$\text{err} = \frac{\lVert g_\text{main} - g_\text{num} \rVert}{\lVert g_\text{main} \rVert + \lVert g_\text{num} \rVert}$$

On attend **err < 1e-6**. Au-dessus de 1e-3, il y a presque sûrement une erreur dans le gradient à la main. Cet outil servira jusqu'en S10 : soignez-le.

---

## 2. Code (≈ 65 min)

Lancez les tests depuis le dossier `S2/` :

```
cd "U:\TIPE TRANSFORMER\S2"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -v
```

Pour essayer une fonction dans IDLE : ouvrez le fichier dans IDLE, puis **F5** (Run Module). Les fonctions sont alors disponibles dans la console.

| Ordre | Fichier | Fonctions | Test associé | Durée |
| --- | --- | --- | --- | --- |
| 1 | `src/regression_logistique.py` | `sigmoide`, `derivee_sigmoide` | `test_sigmoide_*`, `test_derivee_sigmoide` | 10 min |
| 2 | `src/regression_logistique.py` | `probabilites`, `cout`, `gradient` | `test_probabilites_forme`, `test_cout_*`, `test_gradient_valeur_connue` | 15 min |
| 3 | `src/verif_gradient.py` | `gradient_numerique`, `erreur_relative` | `tests/test_verif_gradient.py` | 15 min |
| 4 | `src/regression_logistique.py` | `entrainer` | `test_entrainement_fait_baisser_le_cout` | 5 min |
| 5 | `experiences/s2_couts_eta.py` | `generer_nuages`, `tracer_couts_selon_eta` | la figure elle-même | 20 min |

Chaque fonction contient dans son docstring : ce qu'elle reçoit, ce qu'elle renvoie, les étapes, une valeur de contrôle et les pièges.

---

## 3. Explication croisée et clôture

- Le copilote explique à voix haute, sans regarder l'écran : pourquoi `∂L/∂z = p − y`, et pourquoi le gradient en `w` a la même forme que `w`.
- Remplir `S2/journal.md`, puis commit « S2 » et push.

## 4. Questions possibles du jury

1. Pourquoi ne pas garder la fonction échelon du perceptron ? *Piste : sa dérivée est nulle partout sauf en 0.*
2. Pourquoi l'entropie croisée plutôt que l'erreur quadratique avec une sigmoïde ? *Piste : calculez ∂L/∂z avec $(p-y)^2$ : un facteur p(1−p) apparaît et écrase le gradient quand p est proche de 0 ou 1.*
3. Comment savez-vous que votre gradient est juste ? *Piste : différences finies, erreur relative < 1e-6.*
4. Pourquoi des différences **centrées** plutôt que $\frac{f(w+\varepsilon e_i) - f(w)}{\varepsilon}$ ? *Piste : l'erreur est en O(ε²) au lieu de O(ε) (développement de Taylor).*
