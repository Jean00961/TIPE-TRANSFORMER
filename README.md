# TIPE-TRANSFORMER

Implémentation pas à pas d'un Transformer en Python, avec NumPy uniquement, dans le cadre d'un TIPE (classe de MP). L'objectif est de comprendre et de pouvoir expliquer chaque opération du réseau, pas de reproduire un modèle performant.

## Le projet de TIPE

### Contexte

Les Transformers sont à la base des modèles de langage actuels (GPT et dérivés). Leur élément central est le mécanisme d'attention, qui permet à chaque position d'une séquence de pondérer l'information venant de toutes les autres positions. Le projet s'inscrit dans un thème d'efficacité et d'optimisation : comprendre ce que coûte ce mécanisme et comment il se comporte.

### Objectifs

1. **Comprendre** : reconstruire les opérations d'un Transformer à partir des équations, sans bibliothèque spécialisée, et savoir les expliquer et les justifier.
2. **Implémenter** : écrire en NumPy un petit modèle fonctionnel (forward pass, puis entraînement).
3. **Étudier expérimentalement** : mesurer le comportement du modèle et le confronter aux prédictions théoriques. C'est cette partie qui constitue l'apport scientifique du TIPE, au-delà de la simple reproduction d'un code existant.

### Modèle mathématique

Pour une séquence de `T` vecteurs `X` de dimension `d_model` :

- Projections : `Q = X Wq`, `K = X Wk`, `V = X Wv`
- Matrice d'attention : `A = softmax(Q K^T / sqrt(dk))`, de dimension `T x T`
- Sortie : `Y = A V`
- Multi-head : `h` têtes de dimension `dk = d_model / h`, concaténées puis projetées
- MLP appliqué à chaque position : `W2 ReLU(W1 x + b1) + b2`
- Bloc : attention, connexion résiduelle, LayerNorm, MLP, connexion résiduelle, LayerNorm
- Pour un décodeur de type GPT : masque causal empêchant de regarder les positions futures

### Démarche

1. Représentation des données : tokenisation par caractères, puis matrice d'embedding `|V| x d_model`.
2. Self-attention, testée sur des entrées simples (chaque ligne de `A` somme à 1, effet de la modification d'une entrée sur les sorties).
3. Multi-head attention, avec une boucle sur les têtes dans un premier temps.
4. MLP, bloc Transformer, puis modèle complet produisant une distribution de probabilité sur le vocabulaire.
5. Entraînement : cross-entropy, rétropropagation, optimiseur. Cette étape est traitée à part, car elle représente un chantier distinct de la construction du modèle.
6. Expériences et analyse des résultats.

On part de dimensions minuscules (`d_model = 8`, 2 têtes de dimension 4) avant d'envisager des tailles plus grandes. Le Transformer original utilise `d_model = 512`, 8 têtes et 6 blocs encodeur et décodeur.

### Problématique et axes d'étude

La problématique précise reste à fixer. Le fil conducteur envisagé est le coût de l'attention : pour une séquence de longueur `T`, la matrice `QK^T` est de taille `T x T`, donc le coût de calcul et de mémoire croît quadratiquement avec `T`. Axes d'étude :

- Mesurer le temps d'exécution de l'attention en fonction de `T` et le comparer à la prédiction en `O(T^2)`.
- Étudier le rôle de la normalisation par `sqrt(dk)` (stabilité du softmax).
- Étudier l'effet du nombre de têtes.
- Étudier l'effet de l'encodage positionnel.
- Observer comment la matrice d'attention évolue quand on modifie les entrées.

### Limites assumées

Le but n'est pas de produire un modèle conversationnel performant. Un modèle capable de générer des caractères ressemblant au texte d'entraînement est déjà un résultat suffisant pour mener ces expériences.

### Organisation

Le travail est mené à deux, avec une répartition entre données et architecture d'une part, mécanismes neuronaux d'autre part. L'intégration se fait ensemble et chacun doit pouvoir expliquer l'ensemble du code au jury.

## Approche

On commence par un petit GPT (blocs décodeurs, attention causale) inspiré de la vidéo de Karpathy [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), réécrit en NumPy. On pourra ensuite revenir à l'architecture encodeur-décodeur de *Attention Is All You Need* (2017).

Chaque brique est écrite et testée séparément avant de passer à la suivante, avec des dimensions volontairement minuscules (par exemple `d_model = 8`).

## Structure

```
TIPE-TRANSFORMER/
├── data.py          # tokenisation par caractères (fait)
├── attention.py     # self-attention, multi-head (à venir)
├── mlp.py           # réseau feed-forward (à venir)
├── transformer.py   # bloc et modèle complet (à venir)
└── experiments.ipynb  # expériences et mesures (à venir)
```

## Avancement

- [x] Tokenisation par caractères (`stoi`, `itos`, `encoder`, `decoder`)
- [ ] Matrice d'embedding `|V| x d_model`
- [ ] Self-attention : `Q = XWq`, `K = XWk`, `V = XWv`, `A = softmax(QK^T / sqrt(dk))`, `Y = AV`
- [ ] Multi-head attention
- [ ] MLP : `W2 ReLU(W1 x + b1) + b2`
- [ ] Encodage positionnel, connexions résiduelles, LayerNorm
- [ ] Masque causal et modèle complet
- [ ] Entraînement (fonction de coût, gradients, optimiseur)

## Utilisation

Prérequis : Python 3 et NumPy.

```
python data.py
```

Affiche le vocabulaire, les tokens de "bonjour le monde" et vérifie que le décodage redonne le texte d'origine.

## Pistes d'étude expérimentale

- Évolution de la matrice d'attention selon les entrées.
- Effet du nombre de têtes.
- Rôle de la normalisation par `sqrt(dk)`.
- Effet de l'encodage positionnel.
- Coût de calcul de l'attention en fonction de la longueur de séquence (théoriquement quadratique), comparé à des mesures de temps d'exécution.

## Références

- Vaswani et al., *Attention Is All You Need*, 2017.
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*.
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
