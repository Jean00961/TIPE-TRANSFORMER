# S3 : MLP et rétropropagation manuelle

**Mardi 1er décembre 2026, 2 h. Pilote : Jean. Copilote : Martin.**

**Objectif** : construire un réseau à une couche cachée (MLP) dont le passage avant (*forward*) et le passage arrière (*backward*) sont écrits à la main, vérifier chaque gradient par différences finies, puis résoudre XOR, le problème sur lequel le perceptron de S1 échoue.

**Livrables** : `src/couches.py` et `src/mlp.py` testés (tests vraiment écrits, pas des `pass`), XOR résolu, figure `figures/s3_xor.png`, `journal.md` rempli, commit « S3 » poussé.

Comme pour S2, ce guide ne donne pas le code. Il donne l'ordre, les dimensions, les questions à se poser et des **valeurs de contrôle**. Toutes ont été obtenues avec une solution de référence : si votre fonction redonne la valeur de contrôle, vous êtes sur la bonne voie.

Règle d'or, encore plus importante qu'en S2 : **une couche, puis son test de gradient, puis la suivante.** Un MLP avec une seule erreur de signe dans une couche n'apprend pas, et sans test isolé, on ne sait pas laquelle.

---

## Sommaire

0. [Avant la séance : récupérer les fichiers et faire le point](#0-avant-la-séance--récupérer-les-fichiers-et-faire-le-point)
1. [Rappel actif (10 min)](#1-rappel-actif-010010-seul-sans-notes)
2. [Relance (5 min)](#2-relance-010015)
3. [Théorie (25 min)](#3-théorie-015040-au-tableau)
4. [Code (65 min)](#4-code-040145)
5. [Bonus](#5-bonus-seulement-si-tout-le-reste-est-vert)
6. [Explication croisée (10 min)](#6-explication-croisée-145155)
7. [Clôture (5 min)](#7-clôture-155200)
8. [Fiche-équation n°3](#8-fiche-équation-n3-à-recopier-sur-une-feuille-a5)
9. [Questions possibles du jury](#9-questions-possibles-du-jury)
10. [Plan B](#10-plan-b-si-ça-bloque)
11. [Préparer S4](#11-préparer-s4)

## Déroulé minuté

| Créneau | Contenu | Qui |
| --- | --- | --- |
| 0:00–0:10 | Rappel actif sur S2 (partie 1) | Chacun seul, sans notes |
| 0:10–0:15 | Relance : `pytest` dans `S1/`, `S2/` et `S3/` (partie 2) | Jean |
| 0:15–0:40 | Théorie au tableau (partie 3) | Les deux |
| 0:40–1:45 | Code, tâches 1 à 6 (partie 4) | Jean au clavier, Martin vérifie les dimensions |
| 1:45–1:55 | Explication croisée (partie 6) | Martin |
| 1:55–2:00 | Journal, commit, push (partie 7) | Jean |

Si des reliquats de S2 sont à finir (partie 0.3), ils passent **avant** les tâches de S3 et on coupe dans cet ordre : bonus, puis tâche 6 (figure), puis la fin de la tâche 5 (le test de gradient du MLP entier reste prioritaire).

---

## 0. Avant la séance : récupérer les fichiers et faire le point

### 0.1 Récupérer les fichiers

Tous les fichiers de S3 sont déjà sur GitHub. Dans un terminal PowerShell :

```
cd "U:\TIPE TRANSFORMER"
git pull
```

Vous devez obtenir ce dossier :

```
S3/
├── GUIDE.md                      ← ce guide
├── journal.md                    ← à remplir à la clôture
├── pytest.ini                    ← rend src/ visible pour pytest
├── src/
│   ├── verif_gradient.py         ← à recopier depuis S2, puis à généraliser
│   ├── couches.py                ← Lineaire, ReLU, Sigmoide, EntropieCroiseeBinaire
│   └── mlp.py                    ← MLP, entrainer
├── tests/
│   ├── test_verif_gradient.py
│   ├── test_couches.py
│   └── test_mlp.py
├── experiences/
│   ├── s3_xor.py                 ← figure de la frontière de XOR
│   └── s3_spirale.py             ← bonus
└── figures/
```

Chaque fonction contient dans son docstring : ce qu'elle reçoit, ce qu'elle renvoie, les étapes, une valeur de contrôle et les pièges. **Lisez le docstring en entier avant d'écrire la première ligne.**

### 0.2 Faire le point : où en sont S1 et S2 ?

Attention à un piège : un test dont le corps est `pass` **passe**. Donc « `pytest` affiche 13 passed » ne prouve rien tant que les tests ne sont pas écrits. Pour savoir ce qui reste vraiment à faire, cherchez les `pass` restants :

```
cd "U:\TIPE TRANSFORMER"
Select-String -Path S1\src\*.py, S1\tests\*.py, S2\src\*.py, S2\tests\*.py -Pattern "^\s+pass$"
```

Chaque ligne affichée est une fonction ou un test encore vide.

### 0.3 Ce que S3 réutilise de S2 (indispensable)

S3 a besoin de trois fonctions de S2, **terminées et testées** :

| Fonction de S2 | Où elle sert en S3 |
| --- | --- |
| `sigmoide` (stable) dans `regression_logistique.py` | recopiée dans `couches.py`, pour la couche `Sigmoide` |
| `gradient_numerique` dans `verif_gradient.py` | tous les tests de gradient de S3 |
| `erreur_relative` dans `verif_gradient.py` | idem |

Si elles ne sont pas finies à la fin de S2 : les terminer en **premier** (au plus 25 min, en suivant le guide de S2), avec leurs tests. Le reste de S2 (`entrainer`, figure des courbes selon η) devient un reliquat, noté dans le journal, à faire hors séance.

Les autres reliquats possibles (perceptron et figure de S1, figure de S2) **ne bloquent pas** S3 : on les note dans le journal et on les fait à la maison.

### 0.4 Pourquoi un nouveau dossier ne rend pas S2 inutile

On recopie `verif_gradient.py` et `sigmoide` dans `S3/` pour que S3 tourne tout seul, y compris dans IDLE. La version de `S2/` reste comme trace de la séance. **À partir de S3, la version de référence de `verif_gradient.py` est celle de `S3/`** (elle sait traiter les matrices).

---

## 1. Rappel actif (0:00–0:10, seul, sans notes)

Chacun écrit sur une feuille, de tête, sans regarder ni le code ni le guide de S2. Puis on compare entre nous, puis avec le journal de S2.

1. Écrire la sigmoïde et sa dérivée exprimée avec σ elle-même.
2. Écrire l'entropie croisée binaire pour un exemple, puis le résultat clé : que vaut ∂L/∂z ?
3. Sur un batch : donner ∇w et ∂L/∂b, avec la forme de chaque objet si `X` est de forme `(n, d)`.
4. Écrire la formule des différences finies centrées et le seuil attendu pour l'erreur relative.

<details>
<summary>Réponses (à ouvrir seulement après avoir écrit)</summary>

1. $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ et $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$.
2. $L = -[\,y \ln p + (1-y)\ln(1-p)\,]$ avec $p = \sigma(z)$, et $\dfrac{\partial L}{\partial z} = p - y$.
3. $\nabla_w = \dfrac{1}{n} X^\top (p - y)$, de forme `(d,)` comme `w` ; $\dfrac{\partial L}{\partial b} = \text{moyenne}(p - y)$, un nombre.
4. $\dfrac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}$, erreur relative $\dfrac{\lVert g_1 - g_2\rVert}{\lVert g_1\rVert + \lVert g_2\rVert} < 10^{-6}$.

</details>

Si l'un de nous n'a pas retrouvé ∂L/∂z = p − y : on refait la dérivation au tableau **avant** la théorie de S3 (5 min). Tout S3 repose dessus.

---

## 2. Relance (0:10–0:15)

```
cd "U:\TIPE TRANSFORMER\S2"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -v
cd "U:\TIPE TRANSFORMER\S3"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -v
```

Dans `S2/`, tout doit être vert **et** les tests doivent être écrits (voir 0.2). Dans `S3/`, au départ, tout est « vert » parce que tout est vide : c'est normal.

---

## 3. Théorie (0:15–0:40, au tableau)

On écrit à côté de **chaque** matrice sa forme entre parenthèses. C'est la seule habitude qui évite vraiment les erreurs du backward.

### 3.1 Pourquoi une couche cachée suffit pour XOR (5 min)

En S1, on a montré qu'aucun hyperplan ne sépare XOR. Idée : on combine **deux** hyperplans.

Prenons deux neurones cachés avec la fonction ReLU, $\text{ReLU}(t) = \max(0, t)$ :

$$h_1 = \text{ReLU}(x_1 + x_2), \qquad h_2 = \text{ReLU}(x_1 + x_2 - 1), \qquad s = h_1 - 2\,h_2$$

À faire au tableau : remplir ce tableau.

| $x_1$ | $x_2$ | $h_1$ | $h_2$ | $s$ | XOR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | | | | 0 |
| 0 | 1 | | | | 1 |
| 1 | 0 | | | | 1 |
| 1 | 1 | | | | 0 |

<details>
<summary>Correction</summary>

$h_1$ : 0, 1, 1, 2. $h_2$ : 0, 0, 0, 1. $s$ : 0, 1, 1, 0. On retrouve exactement XOR.

</details>

Lecture géométrique : la couche cachée **change de repère**. Dans le plan $(h_1, h_2)$, les points (0,1) et (1,0) tombent au même endroit, (1, 0), et XOR y devient linéairement séparable. La dernière couche n'est qu'un perceptron dans ce nouveau repère.

Pour obtenir une probabilité, on passe $s$ dans une sigmoïde après l'avoir amplifiée : $p = \sigma(10\,s - 5)$. Ces poids exacts serviront de test en tâche 5 (`test_poids_a_la_main_resolvent_xor`).

Question à se poser : et sans ReLU ? Si $h_1$ et $h_2$ sont linéaires, $s$ est une fonction linéaire de $x$ : la composition de couches linéaires est **linéaire**. Toute la puissance vient de la non-linéarité.

### 3.2 La couche linéaire en lot (3 min)

Pour un lot de $n$ exemples :

$$Z = XW + b$$

| Objet | Forme | Remarque |
| --- | --- | --- |
| `X` | `(n, d_entree)` | une ligne par exemple, comme en S1 et S2 |
| `W` | `(d_entree, d_sortie)` | une **colonne** par neurone de sortie |
| `b` | `(d_sortie,)` | un biais par neurone de sortie |
| `X @ W` | `(n, d_sortie)` | |
| `Z = X @ W + b` | `(n, d_sortie)` | `b` est ajouté à **chaque ligne** (broadcasting) |

Vérification : la colonne $j$ de $W$, c'est le vecteur $w$ du neurone $j$ de S1. Une couche linéaire, c'est $d_\text{sortie}$ neurones de S1 côte à côte, appliqués à $n$ exemples en même temps.

### 3.3 Les trois formules du backward (10 min, la partie la plus importante)

On note $G = \dfrac{\partial L}{\partial Z}$, de forme `(n, d_sortie)` : c'est le gradient qui **arrive** de la couche suivante. La couche doit calculer trois choses :

$$\boxed{\dfrac{\partial L}{\partial W} = X^\top G} \qquad \boxed{\dfrac{\partial L}{\partial b} = \sum_{\text{lignes}} G} \qquad \boxed{\dfrac{\partial L}{\partial X} = G\,W^\top}$$

**Dérivation au tableau, par les indices.** On écrit un coefficient de $Z$ :

$$Z_{ij} = \sum_{k} X_{ik} W_{kj} + b_j$$

1. **Gradient en W.** $W_{kj}$ intervient dans tous les $Z_{ij}$ de la colonne $j$ (pour chaque exemple $i$), avec le coefficient $X_{ik}$. Règle de la chaîne :
   $$\frac{\partial L}{\partial W_{kj}} = \sum_i \frac{\partial L}{\partial Z_{ij}}\,\frac{\partial Z_{ij}}{\partial W_{kj}} = \sum_i G_{ij}\, X_{ik} = \sum_i (X^\top)_{ki}\, G_{ij} = (X^\top G)_{kj}$$
2. **Gradient en b.** $b_j$ intervient dans tous les $Z_{ij}$ de la colonne $j$, avec le coefficient 1 : $\dfrac{\partial L}{\partial b_j} = \sum_i G_{ij}$. C'est la somme de chaque colonne de $G$, soit `G.sum(axis=0)`.
3. **Gradient en X.** $X_{ik}$ intervient dans $Z_{i1}, \dots, Z_{i\,d_\text{sortie}}$ (la ligne $i$), avec le coefficient $W_{kj}$ :
   $$\frac{\partial L}{\partial X_{ik}} = \sum_j G_{ij}\, W_{kj} = \sum_j G_{ij}\,(W^\top)_{jk} = (G\,W^\top)_{ik}$$

**Méthode des dimensions (pour retrouver les formules en 30 s le jour de l'oral).** Le gradient par rapport à un objet a **toujours la forme de cet objet**.

| On veut | Forme attendue | Ingrédients | Seule combinaison possible |
| --- | --- | --- | --- |
| ∂L/∂W | `(d_entree, d_sortie)` | `X (n, d_entree)`, `G (n, d_sortie)` | `X.T @ G` : `(d_entree, n) @ (n, d_sortie)` |
| ∂L/∂X | `(n, d_entree)` | `G (n, d_sortie)`, `W (d_entree, d_sortie)` | `G @ W.T` : `(n, d_sortie) @ (d_sortie, d_entree)` |
| ∂L/∂b | `(d_sortie,)` | `G (n, d_sortie)` | somme sur les `n` lignes |

**Lien avec S2.** En S2, $d_\text{sortie} = 1$ et $G = (p - y)/n$ : la formule $\nabla_w = X^\top(p-y)/n$ est exactement $X^\top G$. Rien de nouveau, c'est la même formule en plus grand.

**Pourquoi la somme sur les lignes pour b ?** Le même biais sert aux $n$ exemples : sa contribution s'additionne. Le $1/n$ de la moyenne n'est pas ici : il est déjà dans $G$, qui vient de la perte (voir 3.5). Piège classique : diviser une deuxième fois par $n$ dans la couche.

**Pourquoi calculer ∂L/∂X alors que X n'est pas un paramètre ?** Parce que le $X$ de la couche 2, c'est la sortie de la couche 1. ∂L/∂X est le $G$ que la couche 2 **transmet** à la couche d'avant. C'est ce qui fait « remonter » le gradient : d'où le nom de rétropropagation.

Exemple numérique à vérifier à la main (c'est aussi la valeur de contrôle de la tâche 2) : avec $X$ = matrice de 1 de forme `(4, 3)`, $W = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$, $b = (0{,}5;\,-0{,}5)$ et $G$ = matrice de 1 de forme `(4, 2)` :
- chaque ligne de $Z$ vaut $(9{,}5;\ 11{,}5)$ ;
- $\partial L/\partial W$ vaut 4 partout (forme `(3, 2)`) ;
- $\partial L/\partial b = (4;\ 4)$ ;
- chaque ligne de $\partial L/\partial X$ vaut $(3;\ 7;\ 11)$.

### 3.4 Les activations : un produit terme à terme (3 min)

Une activation s'applique **coefficient par coefficient** : $A = \varphi(Z)$, même forme que $Z$. Chaque $A_{ij}$ ne dépend que de $Z_{ij}$, donc :

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \varphi'(Z)$$

où $\odot$ est le produit terme à terme (en NumPy : `*`, **pas** `@`).

| Activation | Forward | Ce qu'on met en cache | Backward |
| --- | --- | --- | --- |
| ReLU | $\max(0, Z)$ | le masque `Z > 0` (booléens) | `G * masque` |
| Sigmoïde | $P = \sigma(Z)$ | la sortie `P` | `G * P * (1 - P)` |

La dérivée de ReLU vaut 1 là où $Z > 0$ et 0 ailleurs : le gradient **passe** là où le neurone était actif, et il est **coupé** là où il était éteint. En $Z = 0$, ReLU n'est pas dérivable ; par convention on prend 0 (c'est ce que donne `Z > 0`).

Pour la sigmoïde, on met en cache la sortie $P$ plutôt que $Z$ : grâce à $\sigma' = \sigma(1 - \sigma)$, on n'a pas besoin de recalculer l'exponentielle.

### 3.5 La perte et le retour de « p − y » (3 min)

Pour XOR, le réseau sort $P$ de forme `(n, 1)`, et les étiquettes $Y$ ont la même forme `(n, 1)`.

$$L = -\frac{1}{n}\sum_{i} \big[\,Y_i \ln P_i + (1 - Y_i)\ln(1 - P_i)\,\big] \qquad\Longrightarrow\qquad \frac{\partial L}{\partial P} = \frac{1}{n}\cdot\frac{P - Y}{P\,(1 - P)}$$

(À vérifier : dérivez le terme de l'exemple $i$ par rapport à $P_i$ et mettez au même dénominateur.)

Ce gradient passe ensuite dans la couche `Sigmoide`, qui le multiplie par $P(1-P)$ :

$$\frac{\partial L}{\partial Z} = \frac{1}{n}\cdot\frac{P - Y}{P(1-P)} \odot P(1-P) = \frac{P - Y}{n}$$

**On retrouve le résultat de S2**, mais cette fois **par composition de deux briques indépendantes**. Un test de la tâche 3 vérifie exactement ça (`test_sigmoide_puis_perte_redonne_p_moins_y`).

Piège n°1 de la séance : **la forme de Y**. Si `P` est de forme `(4, 1)` et `y` de forme `(4,)`, alors `P - y` est de forme **`(4, 4)`** (broadcasting) et non `(4, 1)`. Aucune erreur n'est levée : le réseau apprend n'importe quoi en silence. Règle : **Y est toujours de forme `(n, 1)`**, et on écrit `assert P.shape == Y.shape` dans la perte.

Remarque : diviser par $P(1-P)$ devient instable quand $P$ est très proche de 0 ou de 1. On s'en protège ici en bornant $P$ (`np.clip`). En S4, on fusionnera softmax et perte en une seule brique pour supprimer ce problème.

### 3.6 Le graphe de calcul du 2-4-1 (5 min)

Architecture pour XOR : 2 entrées, 4 neurones cachés, 1 sortie. Avec $n = 4$ exemples.

**Passage avant** (de haut en bas). Chaque couche garde en mémoire (cache) ce dont son backward aura besoin.

| Étape | Calcul | Forme | Cache |
| --- | --- | --- | --- |
| entrée | $X$ | `(4, 2)` | |
| `Lineaire(2, 4)` | $Z_1 = XW_1 + b_1$ | `(4, 4)` | $X$ |
| `ReLU` | $H = \max(0, Z_1)$ | `(4, 4)` | masque $Z_1 > 0$ |
| `Lineaire(4, 1)` | $Z_2 = HW_2 + b_2$ | `(4, 1)` | $H$ |
| `Sigmoide` | $P = \sigma(Z_2)$ | `(4, 1)` | $P$ |
| perte | $L$ | nombre | $P$, $Y$ |

**Passage arrière** (de bas en haut : on parcourt la liste des couches **à l'envers**).

| Étape | Reçoit | Calcule et stocke | Renvoie | Forme renvoyée |
| --- | --- | --- | --- | --- |
| perte | | | $\partial L/\partial P$ | `(4, 1)` |
| `Sigmoide` | $\partial L/\partial P$ | | $\partial L/\partial Z_2 = (P - Y)/n$ | `(4, 1)` |
| `Lineaire(4, 1)` | $G_2$ | $dW_2 = H^\top G_2$ `(4, 1)`, $db_2$ `(1,)` | $G_2 W_2^\top$ | `(4, 4)` |
| `ReLU` | $\partial L/\partial H$ | | $\partial L/\partial H \odot (Z_1 > 0)$ | `(4, 4)` |
| `Lineaire(2, 4)` | $G_1$ | $dW_1 = X^\top G_1$ `(2, 4)`, $db_1$ `(4,)` | $G_1 W_1^\top$ (inutilisé) | `(4, 2)` |

À faire au tableau : recopier ce tableau en vérifiant chaque produit matriciel par les formes.

Puis la **mise à jour** : pour chaque couche linéaire, $W \leftarrow W - \eta\, dW$ et $b \leftarrow b - \eta\, db$. Une itération d'entraînement = avant, perte, arrière, mise à jour.

Pourquoi le cache est indispensable : le backward de `Lineaire(4, 1)` a besoin de $H$, calculé pendant le forward. Sans cache, il faudrait tout recalculer. C'est le principe de la rétropropagation : **un seul passage avant, un seul passage arrière**, et on obtient les gradients de **tous** les paramètres pour un coût du même ordre que le forward.

### 3.7 L'initialisation (3 min)

**Pourquoi pas tout à zéro ?** Supposons que toutes les colonnes de $W_1$ soient égales et que tous les coefficients de $b_1$ soient égaux. Alors :
1. les 4 neurones cachés calculent **la même chose** : les 4 colonnes de $H$ sont égales ;
2. dans le backward, ils reçoivent **le même gradient** : les 4 colonnes de $dW_1$ sont égales ;
3. après la mise à jour, les colonnes de $W_1$ sont **encore** égales.

Par récurrence, cela dure pour toujours : les 4 neurones restent des clones, et le réseau se comporte comme s'il n'en avait qu'un. Or un seul neurone caché ne résout pas XOR (un seul hyperplan). Avec des zéros, c'est pire : $H = 0$, donc $dW_2 = H^\top G_2 = 0$ et $dW_1 = 0$ ; rien ne bouge, sauf $b_2$. On le vérifiera (bonus B1).

**Init de He.** On tire $W_{kj} \sim \mathcal{N}\!\left(0,\ \dfrac{2}{d_\text{entree}}\right)$ (variance $2/d_\text{entree}$, donc écart-type $\sqrt{2/d_\text{entree}}$) et $b = 0$. Le hasard casse la symétrie.

D'où vient le 2 ? Si les entrées $x_k$ et les poids sont indépendants et centrés, $\operatorname{Var}(z_j) = d_\text{entree} \cdot \operatorname{Var}(w)\cdot \mathbb{E}[x^2]$. Une ReLU met à zéro environ la moitié des valeurs, donc divise $\mathbb{E}[x^2]$ par 2 à chaque couche. En prenant $\operatorname{Var}(w) = 2/d_\text{entree}$, on compense exactement : l'ordre de grandeur des activations reste le même d'une couche à l'autre, ni explosion ni extinction. (C'est surtout important pour les réseaux profonds ; pour un 2-4-1, toute petite init aléatoire marche à peu près.)

Piège NumPy : `rng.normal(moyenne, ecart_type, size)` prend l'**écart-type**, pas la variance. Il faut passer `np.sqrt(2 / d_entree)`.

### 3.8 Compter les paramètres (1 min)

Une couche `Lineaire(d_entree, d_sortie)` a $d_\text{entree} \times d_\text{sortie}$ poids et $d_\text{sortie}$ biais. Pour le 2-4-1 : $2\times4 + 4 + 4\times1 + 1 = 17$. Pour le futur réseau de S4, 64-128-10 : $64\times128 + 128 + 128\times10 + 10 = 9\,610$. C'est ce nombre qu'on cherchera à réduire par élagage en phase 2.

### 3.9 Comment tester une couche isolée (2 min, indispensable pour la tâche 3)

`gradient_numerique(f, w)` a besoin d'une fonction $f$ qui renvoie **un nombre**. Or une couche renvoie une matrice. L'astuce : on fabrique une perte artificielle avec une matrice $C$ fixe, tirée au hasard, de même forme que la sortie :

$$f(X) = \sum_{i,j} C_{ij}\,\big(\text{couche.avant}(X)\big)_{ij}$$

Pour cette perte, $\partial f / \partial(\text{sortie}) = C$ exactement. Donc :
- le gradient à la main en $X$ s'obtient avec `couche.avant(X)` puis `couche.arriere(C)` ;
- le gradient numérique s'obtient avec `gradient_numerique(lambda V: np.sum(C * couche.avant(V)), X)` ;
- on compare avec `erreur_relative` : on attend moins de 1e-6.

Pour tester $dW$, il faut faire varier `couche.W` à l'intérieur de $f$ : la fonction met la matrice reçue dans `couche.W`, calcule la sortie, **remet l'ancienne valeur**, puis renvoie la somme. Les docstrings de `tests/test_couches.py` détaillent chaque étape.

Pourquoi une matrice $C$ aléatoire et pas une matrice de 1 ? Avec des 1, certaines erreurs se compensent (par exemple une transposée oubliée sur une matrice symétrique). Le hasard rend les coïncidences quasi impossibles.

---

## 4. Code (0:40–1:45)

Lancer les tests depuis `S3/` :

```
cd "U:\TIPE TRANSFORMER\S3"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -v
```

Pour lancer un seul fichier de tests : ajoutez son chemin, par exemple `... -m pytest -v tests/test_couches.py`. Pour un seul test : `... -m pytest -v -k relu`.

Pour essayer une classe dans IDLE : ouvrez `src/couches.py`, **F5**, puis dans la console :

```python
>>> rng = np.random.default_rng(0)
>>> couche = Lineaire(3, 2, rng)
>>> couche.W.shape
(3, 2)
```

### Vue d'ensemble des tâches

| Ordre | Fichier | Contenu | Tests associés | Durée |
| --- | --- | --- | --- | --- |
| 1 | `src/verif_gradient.py` | recopier S2 et généraliser aux matrices | `tests/test_verif_gradient.py` | 10 min |
| 2 | `src/couches.py` | `sigmoide`, `Lineaire` | `test_lineaire_*`, `test_init_he` | 15 min |
| 3 | `src/couches.py` | `ReLU`, `Sigmoide`, `EntropieCroiseeBinaire` | `test_relu_*`, `test_sigmoide_*`, `test_entropie_*` | 10 min |
| 4 | `tests/test_couches.py` | tests de gradient de chaque couche isolée | `test_gradient_*` | 10 min |
| 5 | `src/mlp.py` | `MLP`, `entrainer` | `tests/test_mlp.py` | 15 min |
| 6 | `experiences/s3_xor.py` | figure coût + frontière de XOR | la figure elle-même | 5 min |

Le **minimum vital** de la séance, si tout va mal : tâches 1 à 4 (les couches et leurs tests de gradient). Le MLP n'est qu'un assemblage ; il peut se finir à la maison si les briques sont justes.

### Tâche 1 : `verif_gradient.py` pour les matrices (10 min)

1. Ouvrez `S2/src/verif_gradient.py` et recopiez le **corps** de vos deux fonctions dans `S3/src/verif_gradient.py` (les docstrings de S3 sont déjà là).
2. Problème : en S2, `w` était un vecteur `(d,)`, et vous avez sans doute écrit une boucle `for i in range(len(w))`. Pour une matrice `(3, 2)`, `len(w)` vaut 3 et `w_plus[i]` désigne une **ligne entière** : le résultat est faux.
3. Généralisez avec `np.ndindex(w.shape)`, qui parcourt tous les indices d'un tableau de n'importe quelle forme. Essayez dans IDLE :
   ```python
   >>> for indice in np.ndindex((2, 3)):
   ...     print(indice)
   ```
   Que s'affiche-t-il ? `w_plus[indice] += eps` marche alors pour un vecteur comme pour une matrice.
4. `erreur_relative` marche déjà pour les matrices : `np.linalg.norm` d'une matrice est la racine de la somme des carrés de tous ses coefficients (norme de Frobenius).
5. Écrivez les 4 tests de `tests/test_verif_gradient.py`.

**Valeur de contrôle** : avec `f = lambda M: np.sum(M ** 2)` et `M = np.array([[1., 2., 3.], [4., 5., 6.]])`, `gradient_numerique(f, M)` donne environ `[[2, 4, 6], [8, 10, 12]]`, de forme `(2, 3)`.

### Tâche 2 : `sigmoide` et `Lineaire` (15 min)

1. Recopiez votre `sigmoide` stable de S2 dans `couches.py` (le docstring rappelle les valeurs de contrôle).
2. `Lineaire.__init__(self, d_entree, d_sortie, rng)` : init de He. **Respectez exactement l'appel indiqué dans le docstring**, sinon vos poids ne seront pas les mêmes que ceux de la solution de référence et les valeurs de contrôle de la tâche 5 ne correspondront pas.
3. `Lineaire.avant(self, X)` : mettre `X` en cache (`self.X = X`), renvoyer `Z`.
4. `Lineaire.arriere(self, G)` : calculer et **stocker** `self.dW` et `self.db`, **renvoyer** `dX`.
5. Tests : `test_lineaire_formes`, `test_lineaire_valeurs_a_la_main` (l'exemple de 3.3), `test_init_he`.

**Valeurs de contrôle** : l'exemple numérique de la partie 3.3. Et pour l'init : avec `Lineaire(1000, 500, rng)`, `W.std()` vaut environ 0,0447 ($\sqrt{2/1000}$) et `W.mean()` environ 0.

**Pièges :**
- `self.db = G.sum(axis=0)` et pas `G.sum()` : on veut un vecteur `(d_sortie,)`, pas un nombre. `axis=0` écrase l'axe 0, celui des exemples.
- Écrire `dW = G.T @ X` donne la forme `(d_sortie, d_entree)` : c'est la transposée de ce qu'il faut. Le test de forme le détecte tout de suite.
- `arriere` doit utiliser le `X` **du cache** (celui du dernier `avant`), pas un `X` passé en argument.

### Tâche 3 : `ReLU`, `Sigmoide`, `EntropieCroiseeBinaire` (10 min)

1. `ReLU.avant(Z)` : cache `self.masque = Z > 0`, renvoie `Z * self.masque` (ou `np.maximum(0, Z)`).
2. `ReLU.arriere(G)` : `G * self.masque`.
3. `Sigmoide.avant(Z)` : cache `self.P`, renvoie `self.P`. `Sigmoide.arriere(G)` : `G * P * (1 - P)`.
4. `EntropieCroiseeBinaire.avant(P, Y)` : vérifier les formes, borner `P`, mettre `P` et `Y` en cache, renvoyer la moyenne. `arriere()` : renvoyer ∂L/∂P (formule de 3.5). Attention, `arriere` ne prend **pas** de `G` : la perte est la fin du graphe, le gradient de $L$ par rapport à $L$ vaut 1.
5. Tests : `test_relu_valeurs`, `test_sigmoide_valeurs`, `test_entropie_valeurs`, `test_entropie_refuse_mauvaise_forme`, `test_sigmoide_puis_perte_redonne_p_moins_y`.

**Valeurs de contrôle :**
- ReLU, avec `Z = [[-1, 2], [0, -3]]` : `avant` donne `[[0, 2], [0, 0]]` ; `arriere` d'une matrice de 1 donne `[[0, 1], [0, 0]]` (en 0, la dérivée vaut 0).
- Sigmoide, avec `Z = 0` (matrice `(2, 1)` de zéros) : `avant` donne 0,5 ; `arriere` d'une matrice de 1 donne 0,25.
- Perte, avec `P = [[0.5], [0.5]]` et `Y = [[0.], [1.]]` : $L = \ln 2 \approx 0{,}6931$ ; `arriere()` donne `[[1.], [-1.]]`.
- Perte, avec `P = [[0.9], [0.2]]` et `Y = [[1.], [0.]]` : $L \approx 0{,}1643$.
- Composition : `Sigmoide` sur `Z = 0`, puis la perte avec `Y = [[0.], [1.]]`, puis `sig.arriere(perte.arriere())` donne `[[0.25], [-0.25]]`, c'est-à-dire `(P - Y) / n` avec `n = 2`.

### Tâche 4 : tests de gradient des couches isolées (10 min)

C'est la tâche qui donne confiance dans tout le reste. Relisez la partie 3.9, puis écrivez dans `tests/test_couches.py` :

- `test_gradient_lineaire_en_X`
- `test_gradient_lineaire_en_W`
- `test_gradient_lineaire_en_b`
- `test_gradient_relu`
- `test_gradient_sigmoide`

**Valeur de contrôle** : avec la solution de référence, les erreurs relatives sont de l'ordre de **1e-11**. Si vous obtenez 1e-3 ou plus, il y a une erreur dans le backward. Si vous obtenez autour de 1e-7, vérifiez `eps` (1e-5 attendu).

**Pièges :**
- **L'ordre des calculs.** Calculez d'abord le gradient à la main (`avant` puis `arriere`) et gardez-en une copie (`.copy()`), **puis** le gradient numérique. Pourquoi ? `gradient_numerique` appelle `avant` des dizaines de fois avec des entrées décalées ; après lui, le cache contient le dernier point essayé, pas le vrai point.
- **Restaurer `W`.** Dans la fonction qui fait varier `W`, remettez l'ancienne valeur avant de renvoyer le résultat. Sinon, après `gradient_numerique`, `couche.W` vaut le dernier `W - eps` essayé.
- **ReLU et le point 0.** ReLU n'est pas dérivable en 0. Si un coefficient de `Z` est à moins de `eps` de 0, la différence finie « enjambe » le coude et le test échoue sans que le code soit faux. Avec les `Z` du test (`rng.normal`, graine 0, forme `(5, 4)`), le plus petit `|Z|` vaut environ 0,04, bien plus que `eps = 1e-5` : pas de problème. Mais ne testez jamais ReLU avec des `Z` entiers.

### Tâche 5 : `mlp.py` (15 min)

1. `MLP.__init__(self, tailles, graine=0)` : `tailles = [2, 4, 1]` construit `Lineaire(2, 4)`, `ReLU`, `Lineaire(4, 1)`, `Sigmoide`. Entre deux couches linéaires on met une ReLU ; après la **dernière**, une Sigmoïde. Toutes les couches linéaires utilisent **le même** `rng`, créé une seule fois.
2. `avant(X)` : faire passer `X` dans chaque couche, dans l'ordre.
3. `arriere(G)` : faire passer `G` dans chaque couche, dans l'ordre **inverse** (`reversed(self.couches)`).
4. `maj(eta)` : pour chaque couche linéaire, descendre le gradient. Les activations n'ont pas de paramètres : `isinstance(couche, Lineaire)` permet de les reconnaître.
5. `nombre_parametres()` et `predire(X)`.
6. La fonction `entrainer(mlp, X, Y, eta, n_iterations)`, sur le modèle de S2.
7. Tests de `tests/test_mlp.py`, dans l'ordre du fichier.

**Valeurs de contrôle** (avec `MLP([2, 4, 1], graine=0)` et les données de XOR, `Y` de forme `(4, 1)`) :
- première ligne de `W1` (`mlp.couches[0].W[0]`) : environ `[0.1257, -0.1321, 0.6404, 0.1049]` ;
- `nombre_parametres()` : 17 ;
- `P = mlp.avant(X)` avant entraînement : environ `[[0.5], [0.2951], [0.4154], [0.2628]]`. Le premier vaut exactement 0,5 : pourquoi ? (pensez à $x = (0, 0)$ et $b = 0$) ;
- coût initial : environ 0,7742 ;
- après `entrainer(mlp, X, Y, eta=0.5, n_iterations=2000)` : `couts[100]` ≈ 0,2019, `couts[500]` ≈ 0,0118, `couts[-1]` ≈ 0,0020, et `P` ≈ `[0.006, 0.999, 0.999, 0.001]` ;
- avec les poids à la main de 3.1 (`MLP([2, 2, 1])`, `W1 = [[1, 1], [1, 1]]`, `b1 = [0, -1]`, `W2 = [[10], [-20]]`, `b2 = [-5]`) : `P` ≈ `[0.0067, 0.9933, 0.9933, 0.0067]`. Vérifiez $W_2$ : pourquoi $-20$ et pas $-2$ ? (on a multiplié $s = h_1 - 2h_2$ par 10) ;
- test de gradient du MLP entier (graine 1) : erreur relative ≈ 1e-10.

**Pièges :**
- Si le coût initial n'est pas 0,7742 mais que le reste marche, c'est presque toujours l'appel à `rng.normal` (variance au lieu de l'écart-type, ou `size` dans le mauvais ordre) ou un `rng` recréé pour chaque couche.
- Si le coût **augmente** : signe de la mise à jour (`+=` au lieu de `-=`).
- Si le coût stagne vers 0,69 ($\ln 2$) : regardez la forme de `Y` (piège de 3.5), puis le `reversed`.
- `W -= eta * dW` modifie le tableau **sur place** ; `W = W - eta * dW` crée un nouveau tableau. Les deux marchent ici, mais seule la première forme met à jour `couche.W` si vous travaillez sur un alias local (`W = couche.W`). Écrivez directement `couche.W -= ...`.

### Tâche 6 : la figure de XOR (5 min)

Complétez `experiences/s3_xor.py` (`donnees_xor`, `tracer_frontiere`, `figure_xor`) et lancez-le depuis `S3/` :

```
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" experiences/s3_xor.py
```

La figure `figures/s3_xor.png` a deux panneaux : à gauche la courbe de coût (échelle log), à droite la frontière de décision sur le carré $[-0{,}5;\ 1{,}5]^2$ avec les 4 points. Comparez avec la figure de XOR de S1 : la frontière n'est plus une droite. Combien de « morceaux de droite » voyez-vous ? Reliez ce nombre aux neurones cachés (chaque ReLU ajoute un pli).

---

## 5. Bonus (seulement si tout le reste est vert)

Dans l'ordre de priorité. Les bonus sont la matière première de la problématique : notez chaque résultat surprenant dans la ligne *Étonnement* du journal.

### B1. L'init symétrique, constatée (5 min)

Écrivez `test_init_constante_garde_les_neurones_identiques` (dans `tests/test_mlp.py`) : on remplace tous les poids par 0,5, on entraîne, et on vérifie que les colonnes de `W1` sont toujours toutes égales. Avec la référence, après 2000 itérations, les 4 colonnes sont égales, le coût finit à $\ln 2$ et XOR n'est pas résolu. Avec des poids tous nuls : `W1` reste exactement nul.

### B2. Toutes les graines ne réussissent pas (10 min)

Complétez `taux_de_reussite` et `figure_taux_de_reussite` dans `experiences/s3_xor.py`.

**Avant de lancer**, écrivez dans le journal votre **prédiction** : sur 20 graines, combien de 2-4-1 résolvent XOR ? Et avec 2, 8 ou 16 neurones cachés ? (C'est le réflexe du fil rouge : une hypothèse chiffrée **avant** la mesure.)

<details>
<summary>Ce qu'a donné la solution de référence (à ouvrir après votre mesure)</summary>

- 2-4-1, η = 0,5, 2000 itérations : **15 graines sur 20** résolvent XOR (graines 0 à 19). Les échecs (graines 5, 7, 8, 10, 17) ne dépendent pas de η : on obtient les mêmes avec 0,1 ; 0,5 et 1.
- Sur 20 graines (ce que mesure le script, environ 6 s) : 2 neurones cachés → 0,10 ; 3 → 0,35 ; 4 → 0,75 ; 8 → 1,0 ; 16 → 1,0.
- Sur 50 graines, pour comparer : 2 → 11/50 ; 3 → 25/50 ; 4 → 39/50 ; 8 → 49/50 ; 16 → 50/50. Avec 20 graines seulement, le taux varie de quelques points : c'est une première rencontre avec l'incertitude de mesure.
- Explication de l'échec de la graine 5 : après entraînement, pour 3 des 4 entrées, **les 4 neurones cachés sont éteints** (ReLU à 0). Leur gradient est alors nul (le masque coupe tout) : ces neurones sont « morts » et ne se réveilleront plus. Le réseau sort la même probabilité (environ 0,33) pour ces 3 points et ne peut plus les distinguer.

Observation importante pour la suite du TIPE : **un réseau plus gros que nécessaire s'entraîne mieux**, alors que 2 neurones suffisent en théorie (on l'a montré en 3.1). C'est exactement la question de l'élagage (phase 2) : pourquoi a-t-on besoin de neurones en trop pour apprendre, et peut-on les retirer **après** ? (Voir la *Lottery Ticket Hypothesis*, Frankle et Carbin, 2019, dans la bibliographie du plan.)

</details>

### B3. La spirale (15 min)

Complétez `experiences/s3_spirale.py` : deux spirales imbriquées, impossibles à séparer avec un hyperplan, ni même avec quelques-uns. Comparez trois architectures (η = 0,5, 5000 itérations, graine 0).

| Architecture | Paramètres | Précision | Coût final | Temps (référence) |
| --- | --- | --- | --- | --- |
| 2-16-1 | 65 | 0,775 | ≈ 0,51 | ≈ 0,3 s |
| 2-300-1 | 1201 | 1,0 | ≈ 0,020 | ≈ 2 s |
| 2-32-32-1 | 1185 | 1,0 | ≈ 0,0006 | ≈ 0,6 s |

La comparaison qui compte est celle des deux dernières lignes : **à nombre de paramètres presque égal**, le réseau profond atteint un coût 30 fois plus bas, 3 fois plus vite. Notez-le comme piste : c'est déjà une question de compromis taille/performance, au cœur de la problématique d'élagage.

---

## 6. Explication croisée (1:45–1:55)

Martin explique à voix haute, **sans regarder l'écran**, comme devant le jury. Jean joue le jury et l'interrompt avec ces questions :

1. « Que contient le cache de chaque couche, et pourquoi ? »
2. « Retrouvez ∂L/∂W par les dimensions, sans formule apprise. »
3. « Pourquoi parcourt-on les couches à l'envers dans `arriere` ? »
4. « Comment savez-vous que votre backward est juste ? » (Réponse attendue : un test numérique par couche **isolée**, erreur relative < 1e-6, puis un test sur le réseau entier.)
5. « Pourquoi ne pas initialiser à zéro ? »

Chaque question à laquelle Martin hésite va dans la ligne *Bloqué* du journal : elle sera dans le rappel actif de S4.

---

## 7. Clôture (1:55–2:00)

1. Remplir `S3/journal.md` (5 lignes). La ligne *Première action de la prochaine séance* doit être concrète, par exemple « finir `test_gradient_mlp_entier` » ou « installer scikit-learn et lancer `load_digits` ».
2. Si une tâche est inachevée : laisser un test qui **échoue**, nommé d'après ce qui reste à faire (règle de rattrapage du plan). Il indique où reprendre.
3. Commit et push :

```
cd "U:\TIPE TRANSFORMER"
git add S3
git commit -m "S3 : MLP, couches et rétropropagation testées, XOR résolu"
git push origin main
```

---

## 8. Fiche-équation n°3 (à recopier sur une feuille A5)

**Couche linéaire** : $Z = XW + b$, avec $X$ `(n, d_e)`, $W$ `(d_e, d_s)`, $b$ `(d_s,)`, $Z$ `(n, d_s)`.

**Backward**, avec $G = \partial L/\partial Z$ `(n, d_s)` :
- $\partial L/\partial W = X^\top G$ `(d_e, d_s)`
- $\partial L/\partial b = \sum_\text{lignes} G$ `(d_s,)`
- $\partial L/\partial X = G W^\top$ `(n, d_e)`

**Activations** : $\partial L/\partial Z = \partial L/\partial A \odot \varphi'(Z)$. ReLU : masque $Z > 0$. Sigmoïde : $P(1-P)$.

**Sigmoïde + entropie croisée** : $\partial L/\partial Z = (P - Y)/n$.

**Init de He** : $W \sim \mathcal{N}(0,\ 2/d_e)$, $b = 0$. Jamais de poids tous égaux (symétrie).

**Pièges** : `Y` de forme `(n, 1)`, jamais `(n,)` ; `G.sum(axis=0)` ; ne pas diviser deux fois par $n$ ; écart-type et non variance dans `rng.normal`.

**Phrase à retenir** : « le gradient a la forme de ce qu'on dérive ».

---

## 9. Questions possibles du jury

1. **Pourquoi votre réseau résout-il XOR alors que le perceptron échoue ?** *Piste : la couche cachée combine deux hyperplans ; montrer les poids à la main de 3.1 et le changement de repère $(h_1, h_2)$. Sans non-linéarité, deux couches linéaires n'en font qu'une.*
2. **Comment êtes-vous sûrs de vos gradients ?** *Piste : différences finies centrées, erreur relative < 1e-6, testées couche par couche puis sur le réseau entier ; citer l'ordre de grandeur obtenu (1e-11).*
3. **Pourquoi la rétropropagation plutôt que les différences finies pour entraîner ?** *Piste : les différences finies demandent 2 forwards **par paramètre** (34 pour le 2-4-1, presque 20 000 pour le 64-128-10 de S4) ; la rétropropagation donne tous les gradients en un forward et un backward. Les différences finies ne servent qu'à vérifier.*
4. **Pourquoi ReLU et pas la sigmoïde dans la couche cachée ?** *Piste : $\sigma' \le 1/4$, donc le gradient est divisé par au moins 4 à chaque couche sigmoïde (évanouissement du gradient) ; la dérivée de ReLU vaut 1 là où le neurone est actif. Contrepartie : les neurones morts (bonus B2).*
5. **Pourquoi certaines graines échouent-elles ?** *Piste : neurones ReLU morts, minimum local ; plus de neurones cachés rend l'échec rare. Lien avec la problématique d'élagage : on entraîne grand, puis on élague.*

---

## 10. Plan B (si ça bloque)

- **Un test de gradient échoue et on ne trouve pas pourquoi (plus de 10 min)** : réduire la taille. Prendre `n = 1`, `d_entree = 2`, `d_sortie = 1`, et calculer le gradient à la main sur papier. Comparer coefficient par coefficient avec `print`. Une erreur de transposée saute alors aux yeux.
- **Le test de gradient de la couche passe, mais celui du MLP entier échoue** : l'erreur est dans l'assemblage. Vérifier `reversed`, et que `arriere` de chaque couche reçoit bien la sortie de l'`arriere` de la couche suivante (et pas le `G` de départ).
- **XOR n'est pas résolu alors que tous les tests de gradient passent** : changer de graine (voir B2, certaines échouent normalement). Si aucune graine de 0 à 5 ne marche, vérifier la forme de `Y` et le signe de la mise à jour.
- **Retard de plus de 30 min** : abandonner les bonus et la tâche 6 ; finir au moins les tâches 1 à 4. Le MLP se termine à la maison avant S4.
- **Python ne trouve pas un module** (`ModuleNotFoundError: No module named 'couches'`) : lancer `pytest` depuis `S3/` (là où est `pytest.ini`), pas depuis `S3/tests/`.

---

## 11. Préparer S4

S4 est **bloquante** : le jalon de la phase 1 est un MLP à au moins 95 % en test sur les chiffres manuscrits, avec des gradients vérifiés.

- **Installer scikit-learn avant mardi** (il n'est pas installé sur le poste ; il ne sert qu'à charger les données) :
  ```
  & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pip install scikit-learn
  ```
  Puis vérifier dans IDLE : `from sklearn.datasets import load_digits` puis `load_digits().data.shape` doit donner `(1797, 64)`.
- **Relire la veille (10 min)** : la fiche n°3 et la ligne *Première action* du journal de S3.
- **Ce qui change en S4** : 10 classes au lieu de 2. La sigmoïde devient un **softmax**, et la perte une entropie croisée multiclasse. Résultat à connaître : $\partial L/\partial Z = (P - Y)/n$, **la même formule** qu'aujourd'hui, avec $Y$ en codage one-hot `(n, 10)`. On n'entraîne plus sur tout le jeu à chaque pas, mais par mini-lots de 32.
- **Rappel actif de S4** : réécrire de tête les trois formules du backward avec leurs dimensions.
