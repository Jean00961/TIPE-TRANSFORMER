import { useCallback, useEffect, useState } from "react";

/**
 * État persistant dans localStorage. Le stockage peut être indisponible
 * (navigation privée, données bloquées) : la page fonctionne alors sans mémoire.
 */
export function useStockageLocal<T>(cle: string, initial: T, normaliser: (brut: unknown) => T = (b) => b as T) {
  const [valeur, setValeur] = useState<T>(() => {
    try {
      const brut = localStorage.getItem(cle);
      return brut === null ? initial : normaliser(JSON.parse(brut));
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(cle, JSON.stringify(valeur));
    } catch {
      // Stockage indisponible : rien à faire.
    }
  }, [cle, valeur]);

  return [valeur, setValeur] as const;
}

/** Heure courante, rafraîchie à intervalle régulier. */
export function useHorloge(intervalleMs: number): number {
  const [maintenant, setMaintenant] = useState(() => Date.now());
  useEffect(() => {
    const minuteur = window.setInterval(() => setMaintenant(Date.now()), intervalleMs);
    return () => window.clearInterval(minuteur);
  }, [intervalleMs]);
  return maintenant;
}

/** Séance demandée dans l'adresse (?s=S01), synchronisée avec l'historique du navigateur. */
export function useSeanceDansAdresse() {
  const lire = () => new URLSearchParams(window.location.search).get("s");
  const [demandee, setDemandee] = useState<string | null>(lire);

  useEffect(() => {
    const surRetour = () => setDemandee(lire());
    window.addEventListener("popstate", surRetour);
    return () => window.removeEventListener("popstate", surRetour);
  }, []);

  const aller = useCallback((id: string) => {
    window.history.pushState(null, "", `?s=${id}`);
    setDemandee(id);
    window.scrollTo({ top: 0 });
  }, []);

  return [demandee, aller] as const;
}
