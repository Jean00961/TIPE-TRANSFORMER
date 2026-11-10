// Forme des données de seances/*.json. Toute modification ici doit être reportée
// dans verifier_fiches.py et dans la section « Fiches de séance » de CLAUDE.md.

export type StatutSeance = "faite" | "prochaine" | "a_venir";

export interface Phase {
  numero: number;
  titre: string;
  jalon: string;
}

export interface SeanceResume {
  id: string;
  numero: number;
  titre: string;
  phase: number;
  statut: StatutSeance;
  bloquante: boolean;
  fiche: boolean;
}

export interface IndexSeances {
  binome: [string, string];
  seance_courante: string;
  phases: Phase[];
  seances: SeanceResume[];
}

export interface Creneau {
  debut: number;
  fin: number;
  titre: string;
  qui: string;
  contenu: string;
}

export interface QuestionRappel {
  question: string;
  reponse: string;
}

export interface QuestionJury {
  question: string;
  piste: string;
}

export interface Notion {
  id: string;
  titre: string;
  duree: number;
  au_tableau?: boolean;
  contenu: string;
  formule?: string;
  dimensions?: string;
  piege?: string;
}

export interface Tache {
  id: string;
  titre: string;
  duree: number;
  fichier?: string | null;
  bonus?: boolean;
  etapes: string[];
  test?: string;
  solution?: string;
}

export interface Fiche {
  id: string;
  numero: number;
  titre: string;
  date: string;
  phase: number;
  pilote: string;
  copilote: string;
  genere_le: string;
  objectif: string;
  etat: {
    resume: string;
    reliquats: string[];
    alertes: string[];
  };
  acquis: string[];
  deroule: Creneau[];
  rappel_actif: QuestionRappel[];
  theorie: Notion[];
  taches: Tache[];
  plan_b: string[];
  livrables: string[];
  questions_jury: QuestionJury[];
  ancrage: {
    phrase?: string;
    fiche?: string;
  };
  etonnement?: string;
  suite?: {
    id: string;
    titre: string;
    a_relire?: string;
  };
}

/** Ce que chaque navigateur retient pour une séance (localStorage). */
export interface EtatSeance {
  coches: Record<string, boolean>;
  livrables: Record<string, boolean>;
  debut?: number;
  journal?: string;
}
