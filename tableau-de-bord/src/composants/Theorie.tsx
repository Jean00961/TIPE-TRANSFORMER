import { useEffect, useState } from "react";
import { Blocs, EnLigne, Formule } from "../texte/Texte";
import type { Notion } from "../types";

function CarteNotion({ notion, numero, modeRappel }: { notion: Notion; numero: number; modeRappel: boolean }) {
  const [revelee, setRevelee] = useState(false);
  useEffect(() => setRevelee(false), [modeRappel]);
  const masquee = modeRappel && !revelee;

  return (
    <article className="carte notion">
      <header className="notion__tete">
        <span className="notion__numero">{numero}</span>
        <h3>
          <EnLigne texte={notion.titre} />
        </h3>
        <span className="discret">{notion.duree} min</span>
        {notion.au_tableau && <span className="badge badge--accent">au tableau</span>}
      </header>

      {masquee ? (
        <button type="button" className="bouton notion__reveler" onClick={() => setRevelee(true)}>
          Révéler la notion
        </button>
      ) : (
        <>
          <div className="notion__corps">
            {notion.formule && (
              <div className="notion__formule">
                <Formule tex={notion.formule} bloc />
              </div>
            )}
            <Blocs texte={notion.contenu} />
          </div>
          {notion.dimensions && (
            <div className="encadre">
              <div className="etiquette">Dimensions</div>
              <div>
                <EnLigne texte={notion.dimensions} />
              </div>
            </div>
          )}
          {notion.piege && (
            <div className="encadre encadre--piege">
              <div className="etiquette">Piège</div>
              <div>
                <EnLigne texte={notion.piege} />
              </div>
            </div>
          )}
        </>
      )}
    </article>
  );
}

export function Theorie({ notions, modeRappel }: { notions: Notion[]; modeRappel: boolean }) {
  return (
    <div className="pile">
      {notions.map((n, k) => (
        <CarteNotion key={n.id} notion={n} numero={k + 1} modeRappel={modeRappel} />
      ))}
    </div>
  );
}
