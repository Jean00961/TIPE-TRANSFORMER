# Plan de travail TIPE en 14 séances

En 14 séances de 2 h, on passe du perceptron à un mini-GPT entraîné en NumPy. On élague les deux réseaux. En S14, on a une problématique, des résultats mesurés et un oral déjà répété.

## Vue d'ensemble

| Phase | Séances | Fil conducteur | Jalon de sortie (bloquant) |
| --- | --- | --- | --- |
| 1. Fondations | S1–S4 | Du neurone au MLP entraîné | MLP ≥ 95 % en test, gradients vérifiés |
| 2. Optimisation du MLP | S5–S7 | Élagage et protocole expérimental | Problématique v1, banc d'essai reproductible |
| 3. Transformer | S8–S11 | De l'attention au mini-GPT entraîné | 100 % des gradients vérifiés, texte généré |
| 4. Expériences et oral | S12–S14 | Élagage du Transformer, analyse, oral | Figures finales, diaporama, DOT |

## Déroulé type d'une séance et rituels anti-oubli

Les 14 séances suivent le même découpage. Ce cadre fixe compense la semaine d'écart : la méthode ne change jamais, on n'a à se remettre en tête que le contenu.

| Créneau | Contenu | Qui |
| --- | --- | --- |
| 0:00–0:10 | **Rappel actif** : chacun écrit de tête, sans notes, l'équation clé et 3 idées de la séance précédente, puis compare avec le journal | Chacun seul |
| 0:10–0:15 | **Relance** : `python -m pytest` doit passer en entier avant toute ligne nouvelle | Pilote |
| 0:15–0:40 | **Théorie** : dérivation sur papier, dimensions de chaque matrice écrites à côté | Les deux |
| 0:40–1:45 | **Code** : brique par brique, chaque fonction suivie aussitôt de son test | Pilote au clavier, copilote vérifie les dimensions |
| 1:45–1:55 | **Explication croisée** : le copilote explique le code à voix haute, comme devant le jury | Copilote |
| 1:55–2:00 | **Clôture** : entrée dans `journal.md`, commit, push | Pilote |

**Rituels permanents :**

- **`journal.md`** : toujours les 4 mêmes lignes : *fait*, *compris*, *bloqué*, *première action de la prochaine séance*. La dernière ligne évite de perdre du temps au redémarrage. On ajoute une ligne *étonnement* : c'est la matière première de la problématique.
- **Fiches-équations A5** : une par brique. Chacune contient l'équation, les dimensions et un piège. Elles servent au rappel actif et deviendront les annexes de l'oral.
- **Pilote et copilote alternés** à chaque séance. La répartition du README (données/architecture contre mécanismes neuronaux) vaut pour le travail hors séance. En séance, chacun touche à tout.
- **Relecture de 10 min la veille** : la fiche et la ligne *première action*. Facultatif, mais c'est le meilleur rapport effort/rétention.
- **Les tests servent de mémoire** : on ne commence jamais une brique tant que la précédente n'a pas ses tests.

Arborescence visée dès S1 : `src/` (une brique par fichier), `tests/`, `experiences/` (scripts et CSV), `figures/`, `docs/`, `journal.md`. `src/data.py` (la tokenisation déjà faite) sera réutilisé en S8.

---

## Phase 1 : Fondations

### S1 : Le neurone et le perceptron

1. **Objectif** : écrire et tester un neurone vectorisé et un perceptron qui apprend ET et OU, puis constater qu'il échoue sur XOR.
2. **Théorie (25 min)**
   - Neurone : `y = φ(w·x + b)`. Géométriquement, c'est l'hyperplan `w·x + b = 0`.
   - Règle du perceptron : `w ← w + η (y − ŷ) x`. Citer le théorème de convergence dans le cas séparable.
   - XOR n'est pas linéairement séparable. La preuve tient en 4 inégalités contradictoires, 5 min au tableau.
   - Convention NumPy : `X` de taille `(n, d)`, une ligne par exemple, `X @ w + b` avec broadcasting.
3. **Code**
   1. (15 min) Environnement : venv, numpy, matplotlib, pytest. Créer l'arborescence et `journal.md`.
   2. (15 min) `src/neurone.py` : `neurone(x, w, b)`, puis `neurone_batch(X, w, b)`. Test : la boucle Python et la version vectorisée donnent le même résultat (`np.allclose`).
   3. (20 min) `src/perceptron.py` : `entrainer_perceptron(X, y, eta, n_epoques)` renvoie `w`, `b` et les erreurs par époque. Test : 0 erreur sur ET et sur OU.
   4. (15 min) XOR : les erreurs ne tombent jamais à 0. Tracer la frontière de décision pour ET, OU et XOR avec `contourf`.
4. **Ancrage** : fiche n°1 (neurone, règle, dessin de l'hyperplan). Phrase à retenir : « un neurone = un hyperplan ».
5. **Livrable** : `neurone.py` et `perceptron.py` avec tests verts, `figures/s1_frontieres.png`, commit « S1 ».

### S2 : Apprendre par descente de gradient

1. **Objectif** : minimiser une vraie fonction de coût, avec un gradient calculé à la main et vérifié numériquement.
2. **Théorie**
   - Entropie croisée binaire : `L = −[y ln p + (1−y) ln(1−p)]` avec `p = σ(z)`.
   - `σ'(z) = σ(z)(1 − σ(z))`. Résultat clé : **`∂L/∂z = p − y`**, à dériver au tableau (10 min).
   - Gradient sur un batch : `∇w = Xᵀ(p − y)/n` et `∂b = moyenne(p − y)`.
   - Rôle du pas `η` : trop grand, ça diverge ; trop petit, c'est lent.
   - **Différences finies centrées** : `(L(w+εeᵢ) − L(w−εeᵢ)) / 2ε`. On attend une erreur relative < 1e-6. Cet outil servira jusqu'en S10.
3. **Code**
   1. (10 min) `sigmoide` stable numériquement et sa dérivée. Test : `σ(0) = 0,5`.
   2. (20 min) `src/regression_logistique.py` : forward, coût, gradient.
   3. (15 min) `src/verif_gradient.py` : `gradient_numerique(f, w, eps=1e-5)`, générique, testé sur la régression logistique.
   4. (20 min) Deux nuages gaussiens 2D. Tracer les courbes de coût pour η ∈ {0,01 ; 0,1 ; 1 ; 10} sur une même figure. C'est le premier réflexe expérimental : une seule variable change.
4. **Ancrage** : le rappel actif consiste à redessiner l'hyperplan de S1. Fiche n°2 : « `∂L/∂z = p − y` ».
5. **Livrable** : `verif_gradient.py` testé, figure des courbes de coût selon η.

### S3 : MLP et rétropropagation manuelle

1. **Objectif** : un MLP à une couche cachée, avec forward et backward écrits à la main, qui résout XOR.
2. **Théorie**
   - Couche linéaire : `Z = XW + b`, avec `X (n, d_in)` et `W (d_in, d_out)`.
   - Si `G = ∂L/∂Z` : **`∂L/∂W = XᵀG`**, **`∂L/∂b = Σ_lignes G`**, **`∂L/∂X = GWᵀ`**. On vérifie toujours par les dimensions.
   - ReLU, dont la dérivée est le masque `Z > 0`. Le forward met ses valeurs en cache, le backward parcourt le graphe à l'envers.
   - Initialisation : on n'initialise pas tout à zéro, sinon les neurones restent symétriques. Init de He : `N(0, 2/d_in)`.
   - Pourquoi une couche cachée suffit pour XOR : on compose deux hyperplans.
3. **Code**
   1. (20 min) `src/couches.py` : classes `Lineaire`, `ReLU` et `Sigmoide`, chacune avec `avant(X)` (qui met en cache) et `arriere(G)` (qui stocke `dW` et `db` et renvoie `dX`).
   2. (15 min) Tests de gradient numérique sur chaque couche **isolée**.
   3. (15 min) `src/mlp.py` : `MLP` = liste de couches, avec les méthodes `avant`, `arriere` et `maj(eta)`.
   4. (15 min) XOR avec une architecture 2-4-1. *Bonus :* une spirale générée en NumPy, avec sa frontière non linéaire.
4. **Ancrage** : fiche n°3 (les 3 formules du backward avec leurs dimensions). En S4, le rappel actif consiste à les réécrire de tête.
5. **Livrable** : `couches.py` et `mlp.py` testés, XOR résolu, figure de la frontière.

### S4 : Classification multiclasse sur données réelles

1. **Objectif** : atteindre ≥ 95 % de précision en test sur les chiffres manuscrits 8×8 (`sklearn.datasets.load_digits`, 1797 images). sklearn ne sert qu'au chargement. Un entraînement prend moins de 10 s sur CPU, ce qui sera décisif pour les campagnes d'expériences.
2. **Théorie**
   - Softmax stable : on soustrait le maximum avant l'exponentielle. Avec la cross-entropy et un codage one-hot, **`∂L/∂Z = (P − Y)/n`**.
   - SGD par mini-batch (taille 32), époques, mélange des données.
   - Découpage train/validation/test en 60/20/20, surapprentissage, normalisation des pixels (division par 16).
3. **Code**
   1. (15 min) `SoftmaxCrossEntropie` dans `couches.py`, avec son test numérique.
   2. (15 min) `src/donnees_chiffres.py` : chargement, one-hot, découpage avec une graine fixe.
   3. (25 min) Boucle mini-batch. Historique du coût et de la précision, en train et en validation.
   4. (10 min) Architecture 64-128-10. `np.savez("modeles/mlp_ref.npz")` : c'est **le modèle de référence de toute la phase 2**. Matrice de confusion.
4. **Ancrage, bilan de phase** : chacun explique en 2 min le trajet d'un pixel jusqu'à la prédiction, puis celui d'un gradient jusqu'à un poids. Ce qui bloque devient la priorité du journal.
5. **Livrable** : `mlp_ref.npz`, précision de référence notée dans le journal, courbes d'apprentissage, matrice de confusion.

---

## Phase 2 : Optimisation du MLP et naissance de la problématique

### S5 : Premier élagage et pistes de problématique

1. **Objectif** : mesurer combien de poids on peut retirer au modèle de référence sans perdre en précision, et écrire les premières pistes de problématique.
2. **Théorie**
   - Sur-paramétrisation : l'architecture 64-128-10 compte 9 610 paramètres.
   - Élagage par magnitude non structuré : masque `M = |W| ≥ seuil`, puis `W ← W ⊙ M`, avec un taux d'élagage `s`.
   - **Métriques d'optimisation** : précision, paramètres non nuls, taux de compression, mémoire (format creux CSR), temps d'inférence, FLOPs.
   - Constat qui prépare S6 : une matrice dense à 90 % de zéros ne va pas plus vite en NumPy.
3. **Code**
   1. (20 min) `src/elagage.py` : `masque_magnitude(W, taux)` et `appliquer_masques`. Tests : la proportion de zéros est exacte et les plus grands poids sont conservés.
   2. (20 min) Courbe de la précision en fonction de `s` ∈ {0, 10, …, 90, 95, 98, 99} %, sans réentraînement.
   3. (10 min) Temps d'inférence à s = 0 et à s = 90 % : il ne change pas.
   4. **(25 min) Problématique** : chacun propose seul 3 questions, puis on les met en commun. On note chaque piste selon 3 critères : mesurable avec notre code ? liée au thème de l'année ? faisable en 9 séances ? Exemples :
      - « Dans quelle mesure peut-on réduire le nombre de paramètres d'un réseau sans dégrader ses performances, et quel critère d'élagage est le plus efficace ? »
      - « L'élagage réduit-il réellement le coût de calcul, ou seulement le nombre de paramètres ? »
      - « Toutes les têtes d'attention d'un Transformer sont-elles utiles ? »
4. **Ancrage** : fiche n°5. Résumer la courbe en une phrase : « on retire X % des poids pour moins d'un point de précision perdu ».
5. **Livrable** : `elagage.py` testé, figure précision vs taux, `docs/problematique.md` v0 avec 3 pistes classées.

### S6 : Élagage itératif et structuré

1. **Objectif** : comparer 4 stratégies d'élagage et obtenir un gain de temps **réel**.
2. **Théorie**
   - Réentraînement après élagage : il faut aussi masquer le gradient (`dW ⊙ M`), sinon les poids élagués repoussent.
   - Élagage itératif (Han et al., 2015) : 20 % par étape, puis quelques époques de réentraînement.
   - Seuil global ou seuil par couche : les couches ne sont pas toutes aussi redondantes.
   - **Élagage structuré** : on retire des neurones entiers (colonnes de `W1`, lignes de `W2`). La matrice rétrécit vraiment, donc le temps baisse.
   - En ouverture : l'hypothèse du billet de loterie (Frankle & Carbin, 2019).
3. **Code**
   1. (15 min) Gradient masqué dans `maj`. Test : un poids élagué vaut encore 0 après 10 itérations.
   2. (20 min) Élagage itératif comparé au one-shot.
   3. (15 min) Seuil global contre seuil par couche *(bonus si retard)*.
   4. (15 min) `elaguer_neurones(modele, k)` renvoie un MLP plus petit. Mesure avec `time.perf_counter`, médiane de 100 répétitions sur un batch de 1000.
4. **Ancrage** : redessiner de mémoire la courbe de S5 avant de la regarder. Fiche n°6 : tableau 2×2 (non structuré ou structuré, one-shot ou itératif).
5. **Livrable** : figure avec 4 courbes, tableau temps / précision / paramètres.

### S7 : Protocole expérimental et problématique v1

1. **Objectif** : un banc d'essai reproductible, et une problématique v1 qui alimente la MCOT.
2. **Théorie (ce que le jury évalue)**
   - Graines avec `np.random.default_rng(graine)`. Moyenne ± écart-type sur 5 graines, avec barres d'erreur.
   - Une seule variable à la fois, toujours une baseline.
   - Mesures de temps : échauffement, répétitions, médiane, même machine, rien d'autre qui tourne.
   - Prédiction théorique : une couche linéaire coûte environ `2·n·d_in·d_out` FLOPs. On confronte ce chiffre aux mesures.
   - Une problématique se compose d'une question, d'une hypothèse, d'une grandeur mesurée et d'un critère de réponse.
3. **Code**
   1. (30 min) `experiences/banc.py` : `lancer(config) -> dict`. Chaque essai ajoute une ligne au CSV (la config et les métriques). Le tracé se fait dans un script séparé qui lit le CSV.
   2. (20 min) Relancer S5 et S6 sur 5 graines, figures avec barres d'erreur.
   3. (15 min) Problématique v1 dans `docs/problematique.md`, plus 5 références résumées (titre, idée, lien avec notre travail). C'est la base de la MCOT : positionnement, bibliographie commentée, problématique, objectifs.
4. **Ancrage** : fiche n°7, la **checklist d'expérience** (graine, baseline, 5 essais, une variable, CSV). À relire avant chaque expérience jusqu'en S13.
5. **Livrable** : `banc.py`, CSV, figures avec barres d'erreur, problématique et bibliographie v1. **Vérifier la date limite SCEI de la MCOT** et la noter dans le journal.

---

## Phase 3 : Transformer

### S8 : Embeddings et self-attention

1. **Objectif** : une attention à une tête avec masque causal, testée sur de vraies séquences tokenisées par `data.py`.
2. **Théorie**
   - Embedding : on lit une ligne de `E (|V|, d_model)`, ce qui revient à multiplier un one-hot par `E`. C'est donc une couche linéaire sans biais : lien direct avec S3.
   - `Q = XWq`, `K = XWk`, `V = XWv`, `A = softmax(QKᵀ/√dk)`, `Y = AV`. Chaque ligne de `A` est une distribution sur les positions.
   - Pourquoi `√dk` : la variance de `q·k` est proportionnelle à `dk`. Sans normalisation, le softmax sature et les gradients s'annulent. C'est une piste du README à tester.
   - Masque causal : `−∞` sur le triangle supérieur, avant le softmax.
   - Coût : `O(T²·dk)` en calcul et `T²` en mémoire. Sans encodage positionnel, l'attention est équivariante par permutation.
3. **Code**
   1. (10 min) `src/embedding.py` : forward, et backward avec `np.add.at`.
   2. (25 min) `src/attention.py` et ses tests :
      - chaque ligne de `A` somme à 1 ;
      - avec masque, `A` est triangulaire inférieure ;
      - permuter les entrées permute les sorties ;
      - avec un seul token, `Y = V`.
   3. (15 min) Heatmap de `A` sur « bonjour le monde ».
   4. (15 min) Chronométrage de l'attention pour T de 16 à 1024 avec `banc.py`, tracé en log-log. Noter la pente attendue (≈ 2) **avant** de mesurer.
4. **Ancrage** : fiche n°8 (formule, dimensions `T×d` et `T×T`). Le softmax de S4 est réutilisé tel quel.
5. **Livrable** : `attention.py` testé, heatmap, premier graphe log-log.

### S9 : Multi-head, bloc et modèle complet (forward)

1. **Objectif** : un mini-GPT complet dont le forward produit une distribution sur le vocabulaire.
2. **Théorie**
   - Multi-head : `h` têtes de taille `dk = d_model/h`, concaténées, puis projetées par `Wo`. Le nombre de paramètres reste `4·d_model²`.
   - Encodage positionnel appris `P (T_max, d_model)`, ou sinusoïdal *(bonus)*.
   - LayerNorm (normalisation par ligne, avec `γ` et `β`). Connexion résiduelle `X + f(X)`.
   - Bloc pre-norm à la GPT-2 : `X + MHA(LN(X))`, puis `X + MLP(LN(X))`. Le README décrit la variante post-norm ; choisir et justifier, c'est une question de jury probable.
   - MLP de position `d → 4d → d`, avec les classes `Lineaire` et `ReLU` de S3.
   - Nombre de paramètres ≈ `12·L·d_model² + |V|·d_model`.
3. **Code**
   1. (20 min) Multi-head avec une boucle sur les têtes. Test : `h = 1` redonne l'attention de S8.
   2. (10 min) LayerNorm. Test : moyenne 0 et variance 1 sur chaque ligne.
   3. (20 min) `src/transformer.py` : classes `Bloc` et `MiniGPT`. Config : `d_model = 32`, `h = 4`, `L = 2`, `T = 32`.
   4. (15 min) Tests :
      - dimensions de chaque sortie ;
      - **perte initiale ≈ ln|V|** ;
      - modifier le token t ne change aucune sortie de position < t ;
      - le nombre de paramètres comptés correspond à la formule.
4. **Ancrage** : en début de S10, redessiner le bloc de mémoire. Fiche n°9.
5. **Livrable** : forward complet testé.

### S10 : Rétropropagation du Transformer (séance la plus risquée)

1. **Objectif** : le backward de chaque nouvelle brique, validé par différences finies.
2. **Théorie**
   - Softmax ligne par ligne : `dZ = S ⊙ (G − Σ_ligne(G ⊙ S))`.
   - `Y = AV` : `dA = dY·Vᵀ` et `dV = Aᵀ·dY`. Puis `dQ = dA·K/√dk` et `dK = dAᵀ·Q/√dk`.
   - Les positions masquées ont `S = 0`, donc leur gradient est nul sans traitement spécial.
   - LayerNorm : `dx = (1/σ)(dx̂ − moy(dx̂) − x̂·moy(dx̂ ⊙ x̂))`.
   - Connexion résiduelle : les gradients s'additionnent. Embedding : on accumule avec `np.add.at`.
3. **Code, dans un ordre strict** : une brique n'est commencée que si la précédente est validée.
   1. (15 min) Backward du softmax et son test.
   2. (20 min) Backward de l'attention à une tête et son test.
   3. (15 min) Backward de LayerNorm et son test.
   4. (15 min) Multi-head, bloc, puis modèle complet. Gradient vérifié sur **tous** les paramètres d'un micro-modèle (`d = 8`, `h = 2`, `T = 4`, `L = 1`), en float64, avec une erreur relative < 1e-5.
4. **Ancrage** : fiche n°10 (les 4 backward). Moyen mnémotechnique : « le gradient d'un produit, c'est l'autre facteur transposé, placé pour que les dimensions collent ».
5. **Livrable** : `test_gradients_transformer.py` vert. **En cas de retard, rattrapage obligatoire avant S11.**

### S11 : Entraînement du mini-GPT et problématique finale

1. **Objectif** : entraîner le modèle sur un corpus français, générer du texte et figer la problématique.
2. **Théorie**
   - Adam : moments `m` et `v`, correction de biais. C'est plus robuste que SGD sur un Transformer.
   - Batchs `(B, T)` tirés au hasard, avec une cible décalée d'un caractère.
   - Génération autorégressive, rôle de la température.
   - Corpus de 100 à 500 ko libre de droits, par exemple les Fables de La Fontaine sur Gutenberg.
3. **Code**
   1. (15 min) `src/optimiseurs.py` : SGD et Adam, testés sur une quadratique.
   2. (20 min) Boucle d'entraînement, perte de validation périodique, sauvegarde des poids.
   3. (15 min) Génération, puis lancement de l'entraînement long en arrière-plan (il peut finir à la maison).
   4. **(20 min) Problématique finale** : la phrase, 2 à 3 objectifs, et le plan d'expériences sous forme de tableau (configuration, mesure, hypothèse) pour S12 et S13.
4. **Ancrage** : comparer le texte généré avant et après entraînement. C'est une image mentale forte, à réutiliser à l'oral.
5. **Livrable** : `modeles/gpt_ref.npz`, courbe de perte, 3 échantillons de texte, problématique finale et plan d'expériences.

---

## Phase 4 : Expériences et oral

### S12 : Élagage du Transformer

1. **Objectif** : appliquer au Transformer les outils de la phase 2, en particulier l'élagage de têtes.
2. **Théorie**
   - Importance d'une tête : hausse de la perte quand on masque cette tête (Michel et al., 2019). Alternatives : norme de la sortie, entropie de `A`.
   - Retirer une tête est un élagage structuré, donc un vrai gain de calcul. Les MLP représentent environ 2/3 des paramètres d'un bloc.
3. **Code**
   1. (15 min) Masque de têtes (vecteur de 0 et de 1) et son test.
   2. (25 min) Scores d'importance, retrait tête par tête, courbe de la perte en fonction du nombre de têtes retirées.
   3. (20 min) Élagage par magnitude des MLP contre élagage de l'attention.
   4. (10 min) Temps d'inférence en fonction du nombre de têtes conservées.
4. **Ancrage** : reprendre le tableau 2×2 de S6, appliqué cette fois au Transformer.
5. **Livrable** : CSV, figures, heatmaps des têtes jugées utiles et inutiles.

### S13 : Campagne finale et analyse

1. **Objectif** : les figures définitives qui répondent à la problématique, confrontées à la théorie.
2. **Théorie** : lecture critique des résultats : dispersion entre graines, limites (petit modèle, CPU, NumPy), généralisation. Ce sont les questions favorites du jury.
3. **Pratique**
   1. (40 min) Plan d'expériences sur 5 graines. Les essais longs sont lancés entre S12 et S13.
   2. (30 min) Figures finales : axes titrés, unités, barres d'erreur, **une figure = un message**.
   3. (20 min) `docs/analyse.md` : pour chaque figure, une observation, une interprétation et le lien avec l'hypothèse.
4. **Ancrage** : un pitch d'une minute (problématique et résultat principal), chronométré.
5. **Livrable** : `figures/final/`, tableau de résultats, analyse rédigée.

### S14 : Mise en forme pour le jury

1. **Objectif** : diaporama v1, DOT et oral répété.
2. **Attendus** : vérifier le format exact de l'oral du concours. Structure : contexte, problématique, modèle, méthode, résultats, conclusion. Le DOT se tire directement de `journal.md`. Questions classiques :
   - « Pourquoi √dk ? »
   - « Dérivez le backward du softmax. »
   - « Pourquoi votre élagage n'accélère-t-il pas ? »
   - « Qu'avez-vous fait vous-mêmes ? »
3. **Pratique (pas de nouveau code)**
   1. (45 min) 12 à 15 diapositives.
   2. (20 min) DOT.
   3. (40 min) Deux répétitions chronométrées, puis questions croisées : chacun pose 5 questions à l'autre, tirées des fiches.
   4. (15 min) README à jour, tests verts, tag `v1.0`.
4. **Ancrage** : les fiches-équations deviennent les diapositives d'annexe.
5. **Livrable** : diaporama v1, DOT, tag `v1.0`.

---

## Fil rouge : problématique et apport personnel

Le jury ne valorise pas « avoir codé un Transformer », car des tutoriels existent. Il valorise **des expériences qu'on a conçues** : une hypothèse, des mesures, et leur confrontation à un modèle théorique. Par exemple, la pente 2 prédite pour le temps de l'attention face à la pente mesurée, ou les FLOPs économisés par l'élagage structuré face au temps réellement gagné.

| Séance | Étape | Production |
| --- | --- | --- |
| S1–S4 | Rien à formuler. Noter seulement une ligne *étonnement* par séance | Journal |
| S5 | 3 pistes, classées selon 3 critères | `problematique.md` v0 |
| S6 | Les premières mesures confirment ou écartent des pistes, une piste est retenue | v0.5 |
| S7 | Problématique v1, bibliographie, objectifs | v1, base de la MCOT |
| S8–S9 | Prédictions théoriques écrites **avant** de mesurer | Hypothèses chiffrées |
| S11 | Problématique finale et plan d'expériences | v2 |
| S12–S13 | Résultats et confrontation à la théorie | Figures finales |
| S14 | Conclusion, limites, prolongements | DOT, diaporama |

Axes d'initiative possibles :

- un critère d'importance des têtes de notre invention ;
- une comparaison de la redondance d'un MLP et de celle d'un Transformer ;
- un front de Pareto compression/précision ;
- l'effet de `√dk` sur la stabilité de l'entraînement.

## Règles de rattrapage pendant les vacances

- **S4, S7 et S10 sont bloquantes** : tout retard est rattrapé aux vacances suivantes, avant d'avancer.
- Ce qui peut sauter en cas de retard : les éléments marqués *bonus* (spirale en S3, seuil global contre par couche en S6, encodage sinusoïdal en S9).
- Une séance de rattrapage suit exactement le même format de 2 h, rituels compris.
- Si on doit s'arrêter au milieu d'une brique, on laisse un test qui échoue, nommé d'après la tâche suivante. Il indique où reprendre.

## Bibliographie de départ

Références citées de mémoire : à vérifier avant de les mettre dans la MCOT.

- Vaswani et al., *Attention Is All You Need*, 2017 : [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- LeCun, Denker, Solla, *Optimal Brain Damage*, NeurIPS 1990
- Han et al., *Learning both Weights and Connections for Efficient Neural Networks*, 2015 : [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Frankle & Carbin, *The Lottery Ticket Hypothesis*, 2019 : [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)
- Michel, Levy, Neubig, *Are Sixteen Heads Really Better than One?*, 2019 : [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
- Voita et al., *Analyzing Multi-Head Self-Attention*, 2019 : [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)
- Blalock et al., *What is the State of Neural Network Pruning?*, 2020 : [arXiv:2003.03033](https://arxiv.org/abs/2003.03033)
- Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
