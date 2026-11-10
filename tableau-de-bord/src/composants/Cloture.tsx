import { useState } from "react";
import { journalGenere } from "../progression";
import { EnLigne } from "../texte/Texte";
import type { Fiche } from "../types";

async function copier(texte: string) {
  try {
    await navigator.clipboard.writeText(texte);
    return true;
  } catch {
    return false;
  }
}

export function Cloture({
  fiche,
  coches,
  journal,
  onJournal,
}: {
  fiche: Fiche;
  coches: Record<string, boolean>;
  journal: string | undefined;
  onJournal: (texte: string | undefined) => void;
}) {
  const [copie, setCopie] = useState<"" | "ok" | "echec">("");
  const texte = journal ?? journalGenere(fiche, coches);
  const { ancrage, etonnement, suite } = fiche;

  const surCopie = async () => {
    setCopie((await copier(texte)) ? "ok" : "echec");
    window.setTimeout(() => setCopie(""), 2000);
  };

  return (
    <div className="pile">
      {(ancrage.phrase || ancrage.fiche) && (
        <div className="carte">
          <div className="etiquette">Ancrage</div>
          {ancrage.phrase && (
            <p className="phrase-cle">
              « <EnLigne texte={ancrage.phrase} /> »
            </p>
          )}
          {ancrage.fiche && (
            <p>
              <EnLigne texte={ancrage.fiche} />
            </p>
          )}
          {etonnement && (
            <p className="discret">
              Ligne <em>étonnement</em> : <EnLigne texte={etonnement} />
            </p>
          )}
        </div>
      )}

      <div className="carte">
        <div className="etiquette">Entrée à coller dans journal.md</div>
        <p className="discret">
          Pré-remplie avec les tâches cochées. Complétez les lignes vides, puis copiez. La fiche de la séance suivante est
          préparée à partir de ce journal.
        </p>
        <textarea
          className="journal"
          spellCheck={false}
          value={texte}
          onChange={(e) => onJournal(e.target.value)}
          aria-label="Entrée de journal"
        />
        <div className="boutons">
          <button type="button" className="bouton bouton--principal" onClick={surCopie}>
            {copie === "ok" ? "Copié" : copie === "echec" ? "Copie impossible" : "Copier"}
          </button>
          <button type="button" className="bouton" onClick={() => onJournal(undefined)}>
            Repartir des cases cochées
          </button>
        </div>
      </div>

      {suite && (
        <div className="carte">
          <div className="etiquette">Séance suivante</div>
          <p>
            <strong>{suite.id}</strong> : <EnLigne texte={suite.titre} />
          </p>
          {suite.a_relire && (
            <p>
              <EnLigne texte={suite.a_relire} />
            </p>
          )}
        </div>
      )}
    </div>
  );
}
