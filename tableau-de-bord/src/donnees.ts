// Les fiches sont intégrées au moment du build : chaque push sur main
// fait reconstruire le site par Vercel avec les JSON à jour.

import indexBrut from "../seances/index.json";
import type { Fiche, IndexSeances } from "./types";

const fichesBrutes = import.meta.glob<Fiche>("../seances/S*.json", {
  eager: true,
  import: "default",
});

export const index = indexBrut as IndexSeances;

export const fiches: ReadonlyMap<string, Fiche> = new Map(
  Object.values(fichesBrutes).map((fiche) => [fiche.id, fiche]),
);

/** La séance à afficher : celle demandée si sa fiche existe, sinon la séance courante. */
export function choisirSeance(demandee: string | null): string | undefined {
  if (demandee && fiches.has(demandee)) return demandee;
  if (fiches.has(index.seance_courante)) return index.seance_courante;
  return [...fiches.keys()].sort().at(-1);
}
