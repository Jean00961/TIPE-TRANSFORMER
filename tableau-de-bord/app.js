"use strict";
// Fiches de séance TIPE : lit seances/index.json puis seances/SXX.json et affiche la fiche.
// L'état (tâches cochées, chrono, brouillon du journal) reste dans le navigateur (localStorage).

(function () {
  const contenu = document.getElementById("contenu");
  const frise = document.getElementById("frise");
  const miniChrono = document.getElementById("mini-chrono");
  const caseRappel = document.getElementById("mode-rappel");

  let index = null;
  let fiche = null;
  let etat = {};
  let minuteur = null;

  // ---------- Stockage local (protégé : peut échouer en navigation privée) ----------

  const cle = (id) => "tipe-fiche-" + id;

  function lireStockage(nom, defaut) {
    try {
      const v = JSON.parse(localStorage.getItem(nom));
      return v === null || v === undefined ? defaut : v;
    } catch (e) {
      return defaut;
    }
  }

  function ecrireStockage(nom, valeur) {
    try {
      localStorage.setItem(nom, JSON.stringify(valeur));
    } catch (e) { /* rien : l'affichage fonctionne sans stockage */ }
  }

  function sauver() {
    ecrireStockage(cle(fiche.id), etat);
  }

  // ---------- Mise en forme du texte : Markdown réduit + formules KaTeX ----------

  function echapper(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function formule(tex, display) {
    if (window.katex) {
      try {
        return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: false });
      } catch (e) { /* on retombe sur le texte brut */ }
    }
    return "<code>" + echapper(tex) + "</code>";
  }

  // Texte sur une ligne : `code`, $$maths$$, $maths$, **gras**, *italique*.
  function enLigne(s) {
    if (s === null || s === undefined) return "";
    const reserves = [];
    const garder = (html) => "\u0000" + (reserves.push(html) - 1) + "\u0000";
    let t = String(s);
    t = t.replace(/`([^`]+)`/g, (_, c) => garder("<code>" + echapper(c) + "</code>"));
    t = t.replace(/\$\$([^$]+)\$\$/g, (_, m) => garder(formule(m, true)));
    t = t.replace(/\$([^$]+)\$/g, (_, m) => garder(formule(m, false)));
    t = echapper(t);
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return t.replace(/\u0000(\d+)\u0000/g, (_, i) => reserves[Number(i)]);
  }

  // Texte en blocs : paragraphes, listes "- " et "1. ", blocs de code ```.
  function blocs(texte) {
    if (!texte) return "";
    const lignes = String(texte).split("\n");
    const estPuce = (l) => /^\s*- /.test(l);
    const estNum = (l) => /^\s*\d+\. /.test(l);
    let html = "";
    let i = 0;
    while (i < lignes.length) {
      const l = lignes[i];
      if (l.trim().startsWith("```")) {
        const code = [];
        i++;
        while (i < lignes.length && !lignes[i].trim().startsWith("```")) code.push(lignes[i++]);
        i++;
        html += "<pre><code>" + echapper(code.join("\n")) + "</code></pre>";
      } else if (estPuce(l) || estNum(l)) {
        const test = estPuce(l) ? estPuce : estNum;
        const balise = estPuce(l) ? "ul" : "ol";
        const items = [];
        while (i < lignes.length && test(lignes[i])) items.push(lignes[i++].replace(/^\s*(- |\d+\. )/, ""));
        html += "<" + balise + ">" + items.map((x) => "<li>" + enLigne(x) + "</li>").join("") + "</" + balise + ">";
      } else if (l.trim() === "") {
        i++;
      } else {
        const para = [];
        while (i < lignes.length && lignes[i].trim() !== "" && !lignes[i].trim().startsWith("```") && !estPuce(lignes[i]) && !estNum(lignes[i])) {
          para.push(lignes[i++]);
        }
        html += "<p>" + para.map(enLigne).join("<br>") + "</p>";
      }
    }
    return html;
  }

  // ---------- Outils ----------

  function dateFr(iso) {
    if (!iso) return "";
    const d = new Date(iso + "T12:00:00");
    if (isNaN(d)) return iso;
    return d.toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  }

  function heure(min) {
    const h = Math.floor(min / 60);
    const m = Math.round(min % 60);
    return h + ":" + String(m).padStart(2, "0");
  }

  const liste = (x) => (Array.isArray(x) ? x : []);

  // ---------- Frise des 14 séances ----------

  function rendreFrise() {
    const couleurs = ["--p1", "--p2", "--p3", "--p4"];
    frise.innerHTML = liste(index.phases).map((ph, k) => {
      const seances = liste(index.seances).filter((s) => s.phase === ph.numero);
      const pastilles = seances.map((s) => {
        const classes = ["pastille", s.statut || "a_venir"];
        if (s.bloquante) classes.push("bloquante");
        if (fiche && s.id === fiche.id) classes.push("affichee");
        const bulle = echapper("S" + s.numero + " : " + s.titre + (s.bloquante ? " (bloquante)" : ""));
        return s.fiche
          ? '<a class="' + classes.join(" ") + '" href="?s=' + s.id + '" title="' + bulle + '">S' + s.numero + "</a>"
          : '<span class="' + classes.join(" ") + '" title="' + bulle + ' · fiche pas encore écrite">S' + s.numero + "</span>";
      }).join("");
      return '<div class="phase" style="--couleur-phase: var(' + couleurs[k % 4] + ')">' +
        '<div class="phase-titre" title="Jalon : ' + echapper(ph.jalon) + '">' + ph.numero + ". " + echapper(ph.titre) + "</div>" +
        '<div class="phase-seances">' + pastilles + "</div></div>";
    }).join("");
  }

  // ---------- Fiche ----------

  function tachesObligatoires() {
    return liste(fiche.taches).filter((t) => !t.bonus);
  }

  function rendreFiche() {
    const f = fiche;
    const phase = liste(index.phases).find((p) => p.numero === f.phase) || { titre: "", jalon: "" };
    const coches = etat.coches || {};
    const ref = liste(index.seances).find((s) => s.id === f.id) || {};

    let html = "";

    // En-tête
    html += '<section class="entete" style="margin-top:0">' +
      '<div class="surtitre">Séance ' + f.numero + " / " + liste(index.seances).length + " · Phase " + f.phase + " : " + echapper(phase.titre) +
      (ref.bloquante ? " · séance bloquante" : "") + "</div>" +
      "<h1>" + enLigne(f.titre) + "</h1>" +
      '<div class="meta"><span>' + echapper(dateFr(f.date)) + "</span>" +
      '<span class="puce">Pilote <b>' + echapper(f.pilote || "?") + "</b></span>" +
      '<span class="puce">Copilote <b>' + echapper(f.copilote || "?") + "</b></span></div>" +
      '<p class="objectif">' + enLigne(f.objectif) + "</p>" +
      '<p class="jalon">Jalon de fin de phase : ' + enLigne(phase.jalon) + "</p>" +
      "</section>";

    // Maintenant : chrono + progression
    html += '<div class="carte maintenant" id="maintenant">' +
      '<div><div class="etiquette">Chrono de séance</div>' +
      '<div class="grand-chiffre" id="chrono-temps">–</div>' +
      '<div class="sous-ligne" id="chrono-creneau">Pas encore démarrée</div>' +
      '<div class="jauge"><div id="chrono-jauge" style="width:0"></div></div>' +
      '<div class="boutons"><button class="bouton principal" data-action="demarrer" id="bouton-demarrer">Démarrer la séance</button>' +
      '<button class="bouton discret" data-action="arreter" id="bouton-arreter" hidden>Remettre à zéro</button></div></div>' +
      '<div><div class="etiquette">Reste à coder</div>' +
      '<div class="grand-chiffre" id="reste-nombre">–</div>' +
      '<div class="sous-ligne" id="reste-detail"></div>' +
      '<div class="jauge ok"><div id="reste-jauge" style="width:0"></div></div>' +
      '<div class="sous-ligne" id="reste-retard"></div></div>' +
      "</div>";

    // Où on en est
    const e = f.etat || {};
    html += '<section id="etat"><h2>Où on en est</h2><div class="carte">' + blocs(e.resume);
    if (liste(e.reliquats).length) {
      html += '<div class="reliquat-titre">Reliquats de la séance précédente (à faire en premier)</div><ul>' +
        liste(e.reliquats).map((r) => "<li>" + enLigne(r) + "</li>").join("") + "</ul>";
    }
    html += "</div>" + liste(e.alertes).map((a) => '<div class="alerte">' + enLigne(a) + "</div>").join("") + "</section>";

    // Acquis
    html += '<section id="acquis"><h2>À la fin de la séance, on sait…</h2><div class="carte"><ul class="acquis">' +
      liste(f.acquis).map((a) => "<li>" + enLigne(a) + "</li>").join("") + "</ul></div></section>";

    // Déroulé
    html += '<section id="deroule"><h2>Déroulé <span class="duree">2 h</span></h2><div class="carte" style="padding:6px 0"><ol class="deroule">' +
      liste(f.deroule).map((c, k) => '<li class="creneau" data-creneau="' + k + '">' +
        '<div class="heure">' + heure(c.debut) + "–" + heure(c.fin) + "</div>" +
        '<div><div class="titre-creneau">' + enLigne(c.titre) + ' <span class="qui">· ' + enLigne(c.qui) + "</span></div>" +
        "<p>" + enLigne(c.contenu) + "</p></div></li>").join("") +
      "</ol></div></section>";

    // Rappel actif
    if (liste(f.rappel_actif).length) {
      html += '<section id="rappel"><h2>Rappel actif <span class="duree">de tête, sans notes</span></h2>' +
        liste(f.rappel_actif).map((q, k) => qr(q.question, q.reponse, "rappel-" + k, "Voir la réponse")).join("") + "</section>";
    }

    // Théorie
    const dureeTheorie = liste(f.theorie).reduce((s, t) => s + (t.duree || 0), 0);
    html += '<section id="theorie"><h2>Théorie <span class="duree">' + dureeTheorie + " min</span></h2>" +
      liste(f.theorie).map((t, k) => {
        let c = '<article class="carte notion" id="notion-' + echapper(t.id) + '">' +
          '<div class="notion-tete"><span class="notion-num">' + (k + 1) + "</span><h3>" + enLigne(t.titre) + "</h3>" +
          '<span class="duree">' + (t.duree || 0) + " min</span>" +
          (t.au_tableau ? '<span class="badge tableau">au tableau</span>' : "") + "</div>" +
          '<div class="notion-corps">' +
          (t.formule ? '<div class="formule-cle">' + formule(t.formule, true) + "</div>" : "") +
          blocs(t.contenu) + "</div>";
        if (t.dimensions) c += '<div class="encadre"><div class="etiquette">Dimensions</div><div>' + enLigne(t.dimensions) + "</div></div>";
        if (t.piege) c += '<div class="encadre piege"><div class="etiquette">Piège</div><div>' + enLigne(t.piege) + "</div></div>";
        c += '<button class="bouton bouton-reveler" data-action="reveler-notion">Révéler la notion</button></article>';
        return c;
      }).join("") + "</section>";

    // Tâches de code
    const dureeCode = tachesObligatoires().reduce((s, t) => s + (t.duree || 0), 0);
    html += '<section id="taches"><h2>Code <span class="duree">' + dureeCode + " min hors bonus</span></h2><ol class=\"taches\">" +
      liste(f.taches).map((t) => {
        const fait = !!coches[t.id];
        return '<li class="carte tache' + (fait ? " faite" : "") + '" data-tache="' + echapper(t.id) + '">' +
          '<label class="tache-tete"><input type="checkbox" data-action="cocher" data-id="' + echapper(t.id) + '"' + (fait ? " checked" : "") + ">" +
          '<span class="tache-titre">' + enLigne(t.titre) +
          '<span class="tache-infos"><span class="duree">' + (t.duree || 0) + " min</span>" +
          (t.fichier ? '<span class="mono">' + echapper(t.fichier) + "</span>" : "") +
          (t.bonus ? '<span class="badge bonus">bonus</span>' : "") + "</span></span></label>" +
          '<div class="tache-corps">' +
          (liste(t.etapes).length ? "<ul>" + liste(t.etapes).map((x) => "<li>" + enLigne(x) + "</li>").join("") + "</ul>" : "") +
          (t.test ? '<div class="test">' + enLigne(t.test) + "</div>" : "") +
          (t.solution ? qr("", t.solution, "solution-" + t.id, "Voir la réponse attendue") : "") +
          "</div></li>";
      }).join("") + "</ol></section>";

    // Plan B
    if (liste(f.plan_b).length) {
      html += '<section id="plan-b"><h2>Si on prend du retard</h2><div class="carte"><ul>' +
        liste(f.plan_b).map((p) => "<li>" + enLigne(p) + "</li>").join("") + "</ul></div></section>";
    }

    // Livrables
    const livres = etat.livrables || {};
    html += '<section id="livrables"><h2>Livrables</h2><div class="carte"><ul class="coche-liste">' +
      liste(f.livrables).map((l, k) => '<li><label><input type="checkbox" class="coche" data-action="livrable" data-id="' + k + '"' +
        (livres[k] ? " checked" : "") + "><span>" + enLigne(l) + "</span></label></li>").join("") + "</ul></div></section>";

    // Questions de jury
    if (liste(f.questions_jury).length) {
      html += '<section id="jury"><h2>Questions de jury possibles</h2>' +
        liste(f.questions_jury).map((q, k) => qr(q.question, q.piste, "jury-" + k, "Voir une piste de réponse")).join("") + "</section>";
    }

    // Clôture
    const a = f.ancrage || {};
    html += '<section id="cloture"><h2>Clôture</h2>';
    if (a.phrase || a.fiche) {
      html += '<div class="carte"><div class="etiquette">Ancrage</div>' +
        (a.phrase ? '<p class="phrase-cle">« ' + enLigne(a.phrase) + " »</p>" : "") +
        (a.fiche ? "<p>" + enLigne(a.fiche) + "</p>" : "") +
        (f.etonnement ? '<p class="sous-ligne">Ligne <em>étonnement</em> : ' + enLigne(f.etonnement) + "</p>" : "") + "</div>";
    }
    html += '<div class="carte"><div class="etiquette">Entrée à coller dans journal.md</div>' +
      '<p class="sous-ligne">Pré-remplie avec les tâches cochées. Complétez les lignes vides, puis copiez. La tâche du lundi soir lit ce journal pour préparer la séance suivante.</p>' +
      '<textarea id="journal" spellcheck="false"></textarea>' +
      '<div class="boutons"><button class="bouton principal" data-action="copier" id="bouton-copier">Copier</button>' +
      '<button class="bouton" data-action="regenerer">Repartir des cases cochées</button></div></div>';
    if (f.suite) {
      html += '<div class="carte"><div class="etiquette">Séance suivante</div><p><strong>' + echapper(f.suite.id || "") + "</strong> : " +
        enLigne(f.suite.titre) + "</p>" + (f.suite.a_relire ? "<p>" + enLigne(f.suite.a_relire) + "</p>" : "") + "</div>";
    }
    html += "</section>";

    html += '<p class="pied">Fiche générée le ' + echapper(f.genere_le || "?") + ". Les cases cochées et le chrono restent dans ce navigateur.</p>";

    contenu.innerHTML = html;
    document.title = "S" + f.numero + " · " + f.titre.replace(/[$`*]/g, "");
    majJournal(false);
    majProgression();
    majChrono();
  }

  function qr(question, reponse, id, libelle) {
    return '<div class="carte qr" id="' + id + '">' +
      (question ? '<div class="question">' + enLigne(question) + "</div>" : "") +
      '<button class="bouton discret montrer" data-action="reveler">' + libelle + " ↓</button>" +
      '<div class="reponse">' + blocs(reponse) + "</div></div>";
  }

  // ---------- Progression et chrono ----------

  function minutesFaites() {
    const coches = etat.coches || {};
    return tachesObligatoires().filter((t) => coches[t.id]).reduce((s, t) => s + (t.duree || 0), 0);
  }

  function majProgression() {
    const coches = etat.coches || {};
    const oblig = tachesObligatoires();
    const faites = oblig.filter((t) => coches[t.id]).length;
    const total = oblig.reduce((s, t) => s + (t.duree || 0), 0);
    const reste = total - minutesFaites();
    const el = (id) => document.getElementById(id);
    if (!el("reste-nombre")) return;
    el("reste-nombre").textContent = oblig.length - faites === 0 ? "Tout est fait" : (oblig.length - faites) + " tâche" + (oblig.length - faites > 1 ? "s" : "");
    el("reste-detail").textContent = faites + " / " + oblig.length + " faites · environ " + reste + " min de code restantes";
    el("reste-jauge").style.width = (total ? (100 * (total - reste)) / total : 0) + "%";
    document.querySelectorAll(".tache").forEach((li) => li.classList.toggle("faite", !!coches[li.dataset.tache]));
    majRetard();
  }

  function creneauCode() {
    return liste(fiche.deroule).find((c) => /code/i.test(c.titre)) || null;
  }

  function majRetard() {
    const zone = document.getElementById("reste-retard");
    if (!zone) return;
    zone.textContent = "";
    zone.classList.remove("retard");
    const code = creneauCode();
    if (!etat.debut || !code) return;
    const ecoule = (Date.now() - etat.debut) / 60000;
    if (ecoule < code.debut) return;
    const attendu = Math.min(ecoule, code.fin) - code.debut;
    const ecart = Math.round(attendu - minutesFaites());
    if (ecart >= 10) {
      zone.textContent = "≈ " + ecart + " min de retard sur le code : voir « Si on prend du retard ».";
      zone.classList.add("retard");
    } else if (ecart <= -10) {
      zone.textContent = "≈ " + -ecart + " min d'avance : place pour un bonus.";
    }
  }

  function majChrono() {
    const el = (id) => document.getElementById(id);
    if (!el("chrono-temps")) return;
    const creneaux = document.querySelectorAll(".creneau");
    creneaux.forEach((c) => c.classList.remove("actuel", "passe"));
    if (!etat.debut) {
      el("chrono-temps").textContent = "0:00";
      el("chrono-creneau").textContent = "Pas encore démarrée";
      el("chrono-jauge").style.width = "0";
      el("bouton-demarrer").hidden = false;
      el("bouton-arreter").hidden = true;
      miniChrono.hidden = true;
      return;
    }
    const ecoule = (Date.now() - etat.debut) / 60000;
    const deroule = liste(fiche.deroule);
    const fin = deroule.length ? deroule[deroule.length - 1].fin : 120;
    const k = deroule.findIndex((c) => ecoule >= c.debut && ecoule < c.fin);
    creneaux.forEach((c, i) => {
      if (k === -1 ? ecoule >= fin : i < k) c.classList.add("passe");
      if (i === k) c.classList.add("actuel");
    });
    const actuel = k >= 0 ? deroule[k] : null;
    const texteCreneau = actuel
      ? actuel.titre.replace(/[$`*]/g, "") + " · encore " + Math.max(1, Math.ceil(actuel.fin - ecoule)) + " min"
      : "Séance terminée : clôture";
    el("chrono-temps").textContent = heure(Math.floor(ecoule));
    el("chrono-creneau").textContent = texteCreneau;
    el("chrono-jauge").style.width = Math.min(100, (100 * ecoule) / fin) + "%";
    el("bouton-demarrer").hidden = true;
    el("bouton-arreter").hidden = false;
    miniChrono.hidden = false;
    miniChrono.textContent = heure(Math.floor(ecoule)) + " · " + (actuel ? actuel.titre.replace(/[$`*]/g, "") : "Clôture");
    majRetard();
  }

  // ---------- Journal ----------

  function journalGenere() {
    const coches = etat.coches || {};
    const taches = liste(fiche.taches);
    const faites = taches.filter((t) => coches[t.id]).map((t) => t.titre.replace(/[$*]/g, ""));
    const restantes = tachesObligatoires().filter((t) => !coches[t.id]).map((t) => t.titre.replace(/[$*]/g, ""));
    return "## S" + fiche.numero + " : " + fiche.titre.replace(/[$*]/g, "") + " (" + (fiche.date || "") + ")\n\n" +
      "- Fait : " + (faites.length ? faites.join(" ; ") : "") + "\n" +
      "- Compris : \n" +
      "- Bloqué : " + (restantes.length ? "pas terminé : " + restantes.join(" ; ") : "") + "\n" +
      "- Étonnement : \n" +
      "- Première action de la prochaine séance : \n";
  }

  function majJournal(forcer) {
    const zone = document.getElementById("journal");
    if (!zone) return;
    if (forcer) delete etat.journal;
    zone.value = typeof etat.journal === "string" ? etat.journal : journalGenere();
  }

  // ---------- Événements ----------

  contenu.addEventListener("change", (ev) => {
    const cible = ev.target;
    const action = cible.dataset.action;
    if (action === "cocher") {
      etat.coches = etat.coches || {};
      if (cible.checked) etat.coches[cible.dataset.id] = true;
      else delete etat.coches[cible.dataset.id];
      sauver();
      majProgression();
      if (typeof etat.journal !== "string") majJournal(false);
    } else if (action === "livrable") {
      etat.livrables = etat.livrables || {};
      etat.livrables[cible.dataset.id] = cible.checked;
      sauver();
    }
  });

  contenu.addEventListener("input", (ev) => {
    if (ev.target.id === "journal") {
      etat.journal = ev.target.value;
      sauver();
    }
  });

  contenu.addEventListener("click", (ev) => {
    const bouton = ev.target.closest("[data-action]");
    if (!bouton || bouton.tagName === "INPUT") return;
    const action = bouton.dataset.action;
    if (action === "reveler") {
      bouton.closest(".qr").classList.add("revelee");
    } else if (action === "reveler-notion") {
      bouton.closest(".notion").classList.add("revelee");
    } else if (action === "demarrer") {
      etat.debut = Date.now();
      sauver();
      majChrono();
    } else if (action === "arreter") {
      if (window.confirm("Remettre le chrono à zéro ?")) {
        delete etat.debut;
        sauver();
        majChrono();
      }
    } else if (action === "regenerer") {
      majJournal(true);
      sauver();
    } else if (action === "copier") {
      const zone = document.getElementById("journal");
      const fini = () => {
        bouton.textContent = "Copié";
        setTimeout(() => { bouton.textContent = "Copier"; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(zone.value).then(fini, () => { zone.select(); document.execCommand("copy"); fini(); });
      } else {
        zone.select();
        document.execCommand("copy");
        fini();
      }
    }
  });

  caseRappel.addEventListener("change", () => {
    document.body.classList.toggle("mode-rappel", caseRappel.checked);
    document.querySelectorAll(".notion.revelee").forEach((n) => n.classList.remove("revelee"));
    ecrireStockage("tipe-mode-rappel", caseRappel.checked);
  });

  // ---------- Chargement ----------

  async function lireJson(chemin) {
    const r = await fetch(chemin, { cache: "no-cache" });
    if (!r.ok) throw new Error(chemin + " : " + r.status);
    return r.json();
  }

  async function demarrer() {
    caseRappel.checked = !!lireStockage("tipe-mode-rappel", false);
    document.body.classList.toggle("mode-rappel", caseRappel.checked);
    try {
      index = await lireJson("seances/index.json");
      const demande = new URLSearchParams(location.search).get("s");
      const avecFiche = liste(index.seances).filter((s) => s.fiche).map((s) => s.id);
      const id = avecFiche.includes(demande) ? demande : (avecFiche.includes(index.seance_courante) ? index.seance_courante : avecFiche[avecFiche.length - 1]);
      if (!id) throw new Error("aucune fiche n'est encore écrite");
      fiche = await lireJson("seances/" + id + ".json");
      etat = lireStockage(cle(fiche.id), {});
      if (typeof etat !== "object" || etat === null) etat = {};
      rendreFrise();
      const attendreKatex = () => (window.katex ? Promise.resolve() : new Promise((ok) => {
        let n = 0;
        const t = setInterval(() => { if (window.katex || ++n > 40) { clearInterval(t); ok(); } }, 50);
      }));
      await attendreKatex();
      rendreFiche();
      clearInterval(minuteur);
      minuteur = setInterval(majChrono, 15000);
    } catch (e) {
      const local = location.protocol === "file:";
      contenu.innerHTML = '<div class="erreur"><strong>Impossible de charger la fiche.</strong><p>' + echapper(e.message) + "</p>" +
        (local ? "<p>La page doit être servie par HTTP. Dans le dossier <code>tableau-de-bord</code> : <code>python -m http.server 8000</code>, puis ouvrir <code>http://localhost:8000</code>.</p>" : "") +
        "</div>";
    }
  }

  demarrer();
})();
