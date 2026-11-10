import { ecartCode, finSeance, formaterMinutes, indiceCreneau, minutesDeCode } from "../progression";
import { texteBrut } from "../texte/analyse";
import type { Fiche } from "../types";
import { Jauge } from "./communs";

/** Chrono de la séance et temps de code restant. */
export function PanneauSeance({
  fiche,
  coches,
  minutesEcoulees,
  onDemarrer,
  onRemettreAZero,
}: {
  fiche: Fiche;
  coches: Record<string, boolean>;
  minutesEcoulees: number | null;
  onDemarrer: () => void;
  onRemettreAZero: () => void;
}) {
  const code = minutesDeCode(fiche, coches);
  const restantes = code.nombreTotal - code.nombreFaites;
  const ecart = minutesEcoulees === null ? null : ecartCode(fiche, coches, minutesEcoulees);

  let creneau = "Pas encore démarrée";
  if (minutesEcoulees !== null) {
    const actuel = fiche.deroule[indiceCreneau(fiche, minutesEcoulees)];
    creneau = actuel
      ? `${texteBrut(actuel.titre)} · encore ${Math.max(1, Math.ceil(actuel.fin - minutesEcoulees))} min`
      : "Séance terminée : clôture";
  }

  return (
    <div className="carte panneau">
      <div>
        <div className="etiquette">Chrono de séance</div>
        <div className="grand-chiffre">{formaterMinutes(minutesEcoulees ?? 0)}</div>
        <div className="discret">{creneau}</div>
        <Jauge ratio={(minutesEcoulees ?? 0) / finSeance(fiche)} />
        <div className="boutons">
          {minutesEcoulees === null ? (
            <button type="button" className="bouton bouton--principal" onClick={onDemarrer}>
              Démarrer la séance
            </button>
          ) : (
            <button
              type="button"
              className="bouton bouton--lien"
              onClick={() => window.confirm("Remettre le chrono à zéro ?") && onRemettreAZero()}
            >
              Remettre à zéro
            </button>
          )}
        </div>
      </div>
      <div>
        <div className="etiquette">Reste à coder</div>
        <div className="grand-chiffre">
          {restantes === 0 ? "Tout est fait" : `${restantes} tâche${restantes > 1 ? "s" : ""}`}
        </div>
        <div className="discret">
          {code.nombreFaites} / {code.nombreTotal} faites · environ {code.total - code.faites} min restantes
        </div>
        <Jauge ratio={code.total ? code.faites / code.total : 0} ton="ok" />
        {ecart !== null && ecart >= 10 && (
          <div className="discret discret--alerte">≈ {ecart} min de retard : voir « Si on prend du retard ».</div>
        )}
        {ecart !== null && ecart <= -10 && <div className="discret">≈ {-ecart} min d'avance : place pour un bonus.</div>}
      </div>
    </div>
  );
}
