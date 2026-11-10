def construire_vocabulaire(texte):
    vocabulaire = sorted(set(texte))
    stoi = {c: i for i, c in enumerate(vocabulaire)}
    itos = {i: c for c, i in stoi.items()}
    return vocabulaire, stoi, itos


def encoder(texte, stoi):
    return [stoi[c] for c in texte]


def decoder(tokens, itos):
    return "".join(itos[i] for i in tokens)


def tokeniser(texte):
    vocabulaire, stoi, itos = construire_vocabulaire(texte)
    tokens = encoder(texte, stoi)
    return tokens, vocabulaire, stoi, itos


if __name__ == "__main__":
    texte = "bonjour le monde"
    tokens, vocabulaire, stoi, itos = tokeniser(texte)

    print("Vocabulaire :", vocabulaire)
    print("Taille |V| :", len(vocabulaire))
    print("stoi :", stoi)
    print("Tokens :", tokens)
    print("Décodage :", decoder(tokens, itos))
    assert decoder(tokens, itos) == texte
