# TIPE-TRANSFORMER

Projet de TIPE (MP) : implémentation d'un Transformer en Python/NumPy, brique par brique. Le dépôt est https://github.com/Jean00961/TIPE-TRANSFORMER.

## Git

- Travailler directement sur la branche `main` : pas de branche de fonctionnalité, pas de pull request.
- Toujours pousser sur `origin main` après un commit, sans attendre qu'on le demande.
- Exception : si l'utilisateur demande explicitement de ne pas pousser sur `main` (branche séparée, PR, pas de push), suivre sa demande pour ce cas.
- Cette règle est temporaire ("pour l'instant") et peut être changée par l'utilisateur.
- Ne jamais forcer un push (`--force`) sans demande explicite.

## Conventions

- Langue : français pour les commentaires, noms de fonctions, README et échanges.
- NumPy uniquement pour le calcul (pas de PyTorch pour le forward pass) : l'objectif est de comprendre et reconstruire les opérations.
- Chaque brique (tokenisation, attention, multi-head, MLP, bloc, modèle) doit être testée séparément avant de passer à la suivante.
