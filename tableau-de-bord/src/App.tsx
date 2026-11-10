import { choisirSeance, fiches, index } from "./donnees";
import { useSeanceDansAdresse } from "./hooks";
import { PageSeance } from "./PageSeance";

export function App() {
  const [demandee, aller] = useSeanceDansAdresse();
  const id = choisirSeance(demandee);
  const fiche = id ? fiches.get(id) : undefined;

  if (!fiche) {
    return (
      <main className="page">
        <p className="carte">Aucune fiche de séance n'est encore écrite.</p>
      </main>
    );
  }

  // La clé remonte la page à chaque changement de séance : l'état local est alors relu pour la bonne fiche.
  return <PageSeance key={fiche.id} fiche={fiche} index={index} onChoisir={aller} />;
}
