import { describe, expect, it } from "vitest";
import { fiches, index } from "./donnees";
import { ecartCode, formaterMinutes, indiceCreneau, journalGenere, minutesDeCode } from "./progression";
import type { Fiche } from "./types";

const s01 = fiches.get("S01") as Fiche;

describe("données", () => {
  it("chaque séance marquée fiche=true a son fichier", () => {
    for (const s of index.seances.filter((x) => x.fiche)) expect(fiches.has(s.id)).toBe(true);
  });
});

describe("progression", () => {
  it("compte les minutes de code hors bonus", () => {
    expect(minutesDeCode(s01, {})).toEqual({ total: 65, faites: 0, nombreFaites: 0, nombreTotal: 4 });
    expect(minutesDeCode(s01, { c1: true, c5: true }).faites).toBe(15);
  });

  it("trouve le créneau en cours", () => {
    expect(indiceCreneau(s01, 0)).toBe(0);
    expect(indiceCreneau(s01, 45)).toBe(3);
    expect(indiceCreneau(s01, 130)).toBe(-1);
  });

  it("mesure le retard sur le code", () => {
    expect(ecartCode(s01, {}, 20)).toBeNull();
    expect(ecartCode(s01, {}, 70)).toBe(30);
    expect(ecartCode(s01, { c1: true, c2: true, c3: true }, 60)).toBe(-30);
  });

  it("formate les durées en h:mm", () => {
    expect(formaterMinutes(0)).toBe("0:00");
    expect(formaterMinutes(105.9)).toBe("1:45");
  });

  it("pré-remplit le journal avec les tâches faites et restantes", () => {
    const journal = journalGenere(s01, { c1: true });
    expect(journal).toContain("- Fait : Environnement et arborescence");
    expect(journal).toContain("- Bloqué : pas terminé : Neurone");
  });
});
