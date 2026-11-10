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

/**
 * Catégorie affichée, lue dans l'ancre de l'adresse (#theorie). Une ancre inconnue
 * ou absente donne la première catégorie. Le bouton « retour » du navigateur suit.
 */
export function useCategorieDansAdresse(ids: readonly string[]) {
  const lire = () => {
    const ancre = decodeURIComponent(window.location.hash.slice(1));
    return ids.includes(ancre) ? ancre : (ids[0] ?? "");
  };
  const [active, setActive] = useState<string>(lire);

  useEffect(() => {
    const surChangement = () => setActive(lire());
    window.addEventListener("popstate", surChangement);
    window.addEventListener("hashchange", surChangement);
    return () => {
      window.removeEventListener("popstate", surChangement);
      window.removeEventListener("hashchange", surChangement);
    };
  }, [ids.join(",")]);

  const aller = useCallback((id: string) => {
    if (window.location.hash !== `#${id}`) {
      window.history.pushState(null, "", `${window.location.search}#${id}`);
    }
    setActive(id);
    window.scrollTo({ top: 0 });
  }, []);

  return [active, aller] as const;
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
