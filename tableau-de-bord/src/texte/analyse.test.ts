import { describe, expect, it } from "vitest";
import { decouperBlocs, decouperEnLigne, texteBrut } from "./analyse";

describe("decouperEnLigne", () => {
  it("sépare texte, code, formules, gras et italique", () => {
    expect(decouperEnLigne("a `x @ w` puis $w^*$ et **fort** *doux*")).toEqual([
      { type: "texte", valeur: "a " },
      { type: "code", valeur: "x @ w" },
      { type: "texte", valeur: " puis " },
      { type: "formule", valeur: "w^*", bloc: false },
      { type: "texte", valeur: " et " },
      { type: "gras", valeur: "fort" },
      { type: "texte", valeur: " " },
      { type: "italique", valeur: "doux" },
    ]);
  });

  it("reconnaît une formule en bloc avant une formule en ligne", () => {
    expect(decouperEnLigne("$$y = \\varphi(z)$$")).toEqual([{ type: "formule", valeur: "y = \\varphi(z)", bloc: true }]);
  });

  it("ne lit pas de formule dans du code", () => {
    expect(decouperEnLigne("`prix = $5`")).toEqual([{ type: "code", valeur: "prix = $5" }]);
  });
});

describe("decouperBlocs", () => {
  it("reconnaît paragraphes, listes et blocs de code", () => {
    const texte = "Intro\nsuite\n\n- un\n- deux\n1. premier\n```\nZ = X @ w\n```\nFin";
    expect(decouperBlocs(texte)).toEqual([
      { type: "paragraphe", lignes: ["Intro", "suite"] },
      { type: "liste", ordonnee: false, items: ["un", "deux"] },
      { type: "liste", ordonnee: true, items: ["premier"] },
      { type: "code", texte: "Z = X @ w" },
      { type: "paragraphe", lignes: ["Fin"] },
    ]);
  });
});

describe("texteBrut", () => {
  it("retire le balisage", () => {
    expect(texteBrut("Le pas $\\eta$ sert-il à **quelque chose** ?")).toBe("Le pas \\eta sert-il à quelque chose ?");
  });
});
