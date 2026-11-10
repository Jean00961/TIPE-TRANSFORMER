// Calculs purs sur une fiche : temps de code, créneau en cours, retard, entrée de journal.

import { texteBrut } from "./texte/analyse";
import type { Creneau, Fiche, Tache } from "./types";

export const tachesObligatoires = (fiche: Fiche): Tache[] => fiche.taches.filter((t) => !t.bonus);

const somme = (taches: Tache[]) => taches.reduce((s, t) => s + t.duree, 0);

export function minutesDeCode(fiche: Fiche, coches: Record<string, boolean>) {
  const obligatoires = tachesObligatoires(fiche);
  const faites = obligatoires.filter((t) => coches[t.id]);
  return {
    total: somme(obligatoires),
    faites: somme(faites),
    nombreFaites: faites.length,
    nombreTotal: obligatoires.length,
  };
}

export const creneauCode = (fiche: Fiche): Creneau | undefined =>
  fiche.deroule.find((c) => /code/i.test(c.titre));

export const finSeance = (fiche: Fiche): number => fiche.deroule.at(-1)?.fin ?? 120;

/** Indice du créneau en cours, ou -1 si la séance est terminée. */
export const indiceCreneau = (fiche: Fiche, minutes: number): number =>
  fiche.deroule.findIndex((c) => minutes >= c.debut && minutes < c.fin);

/**
 * Écart entre le code attendu à cette heure et le code coché, en minutes.
 * Positif : retard. Négatif : avance. null avant le créneau de code.
 */
export function ecartCode(fiche: Fiche, coches: Record<string, boolean>, minutes: number): number | null {
  const code = creneauCode(fiche);
  if (!code || minutes < code.debut) return null;
  const attendu = Math.min(minutes, code.fin) - code.debut;
  return Math.round(attendu - minutesDeCode(fiche, coches).faites);
}

export function formaterMinutes(minutes: number): string {
  const total = Math.max(0, Math.floor(minutes));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function journalGenere(fiche: Fiche, coches: Record<string, boolean>): string {
  const faites = fiche.taches.filter((t) => coches[t.id]).map((t) => texteBrut(t.titre));
  const restantes = tachesObligatoires(fiche)
    .filter((t) => !coches[t.id])
    .map((t) => texteBrut(t.titre));
  return [
    `## S${fiche.numero} : ${texteBrut(fiche.titre)} (${fiche.date})`,
    "",
    `- Fait : ${faites.join(" ; ")}`,
    "- Compris : ",
    `- Bloqué : ${restantes.length ? `pas terminé : ${restantes.join(" ; ")}` : ""}`,
    "- Étonnement : ",
    "- Première action de la prochaine séance : ",
    "",
  ].join("\n");
}

export function dateLongue(iso: string): string {
  const date = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
}
