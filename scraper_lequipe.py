"""
Scraper lequipe.fr — 5 derniers matchs d'une équipe de football.

Utilise Playwright (navigateur headless Chromium) pour charger la page
calendrier/résultats de l'équipe sur lequipe.fr et extraire les matchs terminés.

URL cible : https://www.lequipe.fr/Football/{competition}/page-calendrier-general/{slug}

Usage :
    python3 scraper_lequipe.py "PSG"
    python3 scraper_lequipe.py "Olympique de Marseille" -c ligue-1
    python3 scraper_lequipe.py "Arsenal" -c championnat-d-angleterre
    python3 scraper_lequipe.py "Bayern" -c bundesliga
    python3 scraper_lequipe.py --list-slugs -c ligue-1     # affiche slugs disponibles
    python3 scraper_lequipe.py "PSG" -o psg_stats.json

Sortie JSON compatible avec l'app Streamlit :
    {
      "team_name":          "Paris-SG",
      "source":             "lequipe.fr",
      "last_updated":       "2026-04-22T...",
      "nb_matches":         5,
      "wins":               4,
      "draws":              0,
      "losses":             1,
      "goals_scored_avg":   2.4,
      "goals_conceded_avg": 0.8,
      "goals_scored":       2.4,   ← alias pour app.py
      "goals_conceded":     0.8,   ← alias pour app.py
      "matches": [...]
    }
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Auto-installation de Chromium (nécessaire sur Streamlit Community Cloud)
# ---------------------------------------------------------------------------

def _ensure_chromium() -> None:
    """
    Vérifie que le binaire Chromium de Playwright est disponible.
    Si absent (ex. premier démarrage sur Streamlit Cloud), le télécharge
    silencieusement via `playwright install chromium`.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as _p:
            exe = _p.chromium.executable_path
            if not os.path.exists(exe):
                raise FileNotFoundError(exe)
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=False,
            capture_output=True,
        )


_ensure_chromium()

# ---------------------------------------------------------------------------
# Compétitions disponibles sur lequipe.fr
# ---------------------------------------------------------------------------
COMPETITIONS = {
    "ligue-1":                    "Ligue 1",
    "ligue-2":                    "Ligue 2",
    "ligue-des-champions":        "Ligue des Champions",
    "ligue-europa":               "Ligue Europa",
    "ligue-conference":           "Ligue Europa Conférence",
    "coupe-de-france":            "Coupe de France",
    "championnat-d-angleterre":   "Premier League",
    "championnat-d-espagne":      "La Liga",
    "championnat-d-italie":       "Serie A",
    "championnat-d-allemagne":    "Bundesliga",
    "championnat-du-portugal":    "Primeira Liga",
    "coupe-intercontinentale":    "Coupe Intercontinentale",
    "super-coupe-d-europe":       "Super Coupe d'Europe",
    "trophee-des-champions":      "Trophée des Champions",
    # Coupes nationales
    "coupe-d-allemagne":          "Coupe d'Allemagne",
    "fa-cup":                     "FA Cup",
    "league-cup":                 "League Cup",
    "coupe-du-roi":               "Copa del Rey",
    "coupe-d-italie":             "Coupe d'Italie",
}

# Compétitions supplémentaires à scraper selon la ligue principale
# (en plus du championnat national, pour couvrir toutes les compétitions)
EXTRA_COMPS_BY_MAIN: dict[str, list[str]] = {
    "ligue-1":                  ["ligue-des-champions", "ligue-europa", "ligue-conference", "coupe-de-france"],
    "championnat-d-allemagne":  ["ligue-des-champions", "ligue-europa", "ligue-conference", "coupe-d-allemagne"],
    "championnat-d-angleterre": ["ligue-des-champions", "ligue-europa", "ligue-conference", "fa-cup", "league-cup"],
    "championnat-d-espagne":    ["ligue-des-champions", "ligue-europa", "ligue-conference", "coupe-du-roi"],
    "championnat-d-italie":     ["ligue-des-champions", "ligue-europa", "ligue-conference", "coupe-d-italie"],
}

# Durée de vie du cache (heures) — données considérées fraîches pendant cette durée
CACHE_TTL_HOURS = 6

# Compétitions par défaut à essayer selon le nom de l'équipe
DEFAULT_COMPETITIONS_FR = ["ligue-1", "ligue-des-champions"]
DEFAULT_COMPETITIONS_EN = ["championnat-d-angleterre", "ligue-des-champions"]
DEFAULT_COMPETITIONS_ES = ["championnat-d-espagne", "ligue-des-champions"]
DEFAULT_COMPETITIONS_DE = ["championnat-d-allemagne", "ligue-des-champions"]
DEFAULT_COMPETITIONS_IT = ["championnat-d-italie", "ligue-des-champions"]

# ---------------------------------------------------------------------------
# Table de correspondance noms → slugs lequipe.fr (enrichie à la volée)
# ---------------------------------------------------------------------------
# Slug → compétition principale sur lequipe.fr
# Slugs vérifiés directement sur lequipe.fr via les pages club
SLUG_COMPETITION: dict[str, str] = {
    # Ligue 1 (18 équipes 2025-26)
    "paris-sg": "ligue-1", "marseille": "ligue-1", "lyon": "ligue-1",
    "monaco": "ligue-1", "lille": "ligue-1", "nice": "ligue-1",
    "rennes": "ligue-1", "lens": "ligue-1", "strasbourg": "ligue-1",
    "nantes": "ligue-1", "brest": "ligue-1", "toulouse": "ligue-1",
    "le-havre": "ligue-1", "lorient": "ligue-1", "metz": "ligue-1",
    "auxerre": "ligue-1", "angers": "ligue-1", "paris-fc": "ligue-1",
    # Premier League (20 équipes 2025-26)
    "arsenal": "championnat-d-angleterre",
    "manchester-city": "championnat-d-angleterre",
    "manchester-united": "championnat-d-angleterre",
    "liverpool": "championnat-d-angleterre",
    "chelsea": "championnat-d-angleterre",
    "tottenham": "championnat-d-angleterre",
    "aston-villa": "championnat-d-angleterre",
    "newcastle": "championnat-d-angleterre",
    "brighton": "championnat-d-angleterre",
    "everton": "championnat-d-angleterre",
    "brentford": "championnat-d-angleterre",
    "fulham": "championnat-d-angleterre",
    "wolverhampton": "championnat-d-angleterre",
    "crystal-palace": "championnat-d-angleterre",
    "nottingham-forest": "championnat-d-angleterre",
    "bournemouth": "championnat-d-angleterre",
    "west-ham": "championnat-d-angleterre",
    "burnley": "championnat-d-angleterre",
    "leeds": "championnat-d-angleterre",
    "sunderland": "championnat-d-angleterre",
    # La Liga (20 équipes 2025-26)
    "real-madrid": "championnat-d-espagne",
    "fc-barcelone": "championnat-d-espagne",
    "atletico-de-madrid": "championnat-d-espagne",
    "seville-fc": "championnat-d-espagne",
    "villarreal": "championnat-d-espagne",
    "real-sociedad": "championnat-d-espagne",
    "athletic-bilbao": "championnat-d-espagne",
    "betis-seville": "championnat-d-espagne",
    "valence-cf": "championnat-d-espagne",
    "osasuna": "championnat-d-espagne",
    "getafe": "championnat-d-espagne",
    "rayo-vallecano": "championnat-d-espagne",
    "celta-vigo": "championnat-d-espagne",
    "alaves": "championnat-d-espagne",
    "espanyol-barcelone": "championnat-d-espagne",
    "gerone": "championnat-d-espagne",
    "majorque": "championnat-d-espagne",
    "elche": "championnat-d-espagne",
    "levante": "championnat-d-espagne",
    "oviedo": "championnat-d-espagne",
    # Bundesliga (18 équipes 2025-26)
    "bayern-munich": "championnat-d-allemagne",
    "borussia-dortmund": "championnat-d-allemagne",
    "bayer-leverkusen": "championnat-d-allemagne",
    "rb-leipzig": "championnat-d-allemagne",
    "eintracht-francfort": "championnat-d-allemagne",
    "wolfsburg": "championnat-d-allemagne",
    "vfb-stuttgart": "championnat-d-allemagne",
    "union-berlin": "championnat-d-allemagne",
    "borussia-m-gladbach": "championnat-d-allemagne",
    "fc-cologne": "championnat-d-allemagne",
    "fribourg": "championnat-d-allemagne",
    "werder-breme": "championnat-d-allemagne",
    "augsbourg": "championnat-d-allemagne",
    "hoffenheim": "championnat-d-allemagne",
    "mayence": "championnat-d-allemagne",
    "heidenheim": "championnat-d-allemagne",
    "sankt-pauli": "championnat-d-allemagne",
    "hambourg-sv": "championnat-d-allemagne",
    # Serie A (20 équipes 2025-26)
    "juventus-turin": "championnat-d-italie",
    "inter-milan": "championnat-d-italie",
    "ac-milan": "championnat-d-italie",
    "naples": "championnat-d-italie",
    "as-rome": "championnat-d-italie",
    "lazio-rome": "championnat-d-italie",
    "atalanta-bergame": "championnat-d-italie",
    "fiorentina": "championnat-d-italie",
    "torino": "championnat-d-italie",
    "bologne": "championnat-d-italie",
    "genoa": "championnat-d-italie",
    "udinese": "championnat-d-italie",
    "cagliari": "championnat-d-italie",
    "hellas-verone": "championnat-d-italie",
    "come": "championnat-d-italie",
    "lecce": "championnat-d-italie",
    "parme": "championnat-d-italie",
    "sassuolo": "championnat-d-italie",
    "cremonese": "championnat-d-italie",
    "pise": "championnat-d-italie",
}

KNOWN_SLUGS: dict[str, str] = {
    # Ligue 1
    "psg": "paris-sg",
    "paris saint-germain": "paris-sg",
    "paris sg": "paris-sg",
    "marseille": "marseille",
    "om": "marseille",
    "olympique de marseille": "marseille",
    "lyon": "lyon",
    "ol": "lyon",
    "olympique lyonnais": "lyon",
    "monaco": "monaco",
    "as monaco": "monaco",
    "lille": "lille",
    "losc": "lille",
    "nice": "nice",
    "ogc nice": "nice",
    "rennes": "rennes",
    "stade rennais": "rennes",
    "lens": "lens",
    "rc lens": "lens",
    "strasbourg": "strasbourg",
    "nantes": "nantes",
    "fc nantes": "nantes",
    "brest": "brest",
    "stade brestois": "brest",
    "montpellier": "montpellier",
    "toulouse": "toulouse",
    "reims": "reims",
    "le havre": "le-havre",
    "lorient": "lorient",
    "metz": "metz",
    "auxerre": "auxerre",
    "angers": "angers",
    "paris fc": "paris-fc",
    "saint-etienne": "saint-etienne",
    "asse": "saint-etienne",
    "bordeaux": "bordeaux",
    "clermont": "clermont",
    # Premier League
    "arsenal": "arsenal",
    "manchester city": "manchester-city",
    "man city": "manchester-city",
    "manchester united": "manchester-united",
    "man united": "manchester-united",
    "man utd": "manchester-united",
    "liverpool": "liverpool",
    "chelsea": "chelsea",
    "tottenham": "tottenham",
    "spurs": "tottenham",
    "aston villa": "aston-villa",
    "newcastle": "newcastle",
    "newcastle united": "newcastle",
    "brighton": "brighton",
    "everton": "everton",
    "brentford": "brentford",
    "fulham": "fulham",
    "wolves": "wolverhampton",
    "wolverhampton": "wolverhampton",
    "crystal palace": "crystal-palace",
    "nottingham forest": "nottingham-forest",
    "nottm forest": "nottingham-forest",
    "bournemouth": "bournemouth",
    "west ham": "west-ham",
    "west ham united": "west-ham",
    "burnley": "burnley",
    "leeds": "leeds",
    "leeds united": "leeds",
    "sunderland": "sunderland",
    # La Liga
    "real madrid": "real-madrid",
    "barcelona": "fc-barcelone",
    "fc barcelona": "fc-barcelone",
    "fc barcelone": "fc-barcelone",
    "barca": "fc-barcelone",
    "atletico madrid": "atletico-de-madrid",
    "atletico": "atletico-de-madrid",
    "atleti": "atletico-de-madrid",
    "sevilla": "seville-fc",
    "seville": "seville-fc",
    "fc seville": "seville-fc",
    "villarreal": "villarreal",
    "real sociedad": "real-sociedad",
    "athletic bilbao": "athletic-bilbao",
    "athletic club": "athletic-bilbao",
    "athletic": "athletic-bilbao",
    "real betis": "betis-seville",
    "betis": "betis-seville",
    "betis seville": "betis-seville",
    "valencia": "valence-cf",
    "valence": "valence-cf",
    "osasuna": "osasuna",
    "ca osasuna": "osasuna",
    "getafe": "getafe",
    "getafe cf": "getafe",
    "rayo vallecano": "rayo-vallecano",
    "rayo": "rayo-vallecano",
    "celta vigo": "celta-vigo",
    "celta": "celta-vigo",
    "rc celta": "celta-vigo",
    "alaves": "alaves",
    "deportivo alaves": "alaves",
    "espanyol": "espanyol-barcelone",
    "rcd espanyol": "espanyol-barcelone",
    "girona": "gerone",
    "girona fc": "gerone",
    "mallorca": "majorque",
    "rcd mallorca": "majorque",
    "elche": "elche",
    "elche cf": "elche",
    "levante": "levante",
    "levante ud": "levante",
    "oviedo": "oviedo",
    "real oviedo": "oviedo",
    # Bundesliga
    "bayern": "bayern-munich",
    "bayern munich": "bayern-munich",
    "fc bayern": "bayern-munich",
    "fc bayern munich": "bayern-munich",
    "borussia dortmund": "borussia-dortmund",
    "dortmund": "borussia-dortmund",
    "bvb": "borussia-dortmund",
    "bayer leverkusen": "bayer-leverkusen",
    "leverkusen": "bayer-leverkusen",
    "rb leipzig": "rb-leipzig",
    "leipzig": "rb-leipzig",
    "eintracht frankfurt": "eintracht-francfort",
    "frankfurt": "eintracht-francfort",
    "francfort": "eintracht-francfort",
    "wolfsburg": "wolfsburg",
    "vfl wolfsburg": "wolfsburg",
    "freiburg": "fribourg",
    "sc freiburg": "fribourg",
    "hoffenheim": "hoffenheim",
    "tsg hoffenheim": "hoffenheim",
    "mainz": "mayence",
    "mainz 05": "mayence",
    "stuttgart": "vfb-stuttgart",
    "vfb stuttgart": "vfb-stuttgart",
    "union berlin": "union-berlin",
    "1 fc union berlin": "union-berlin",
    "gladbach": "borussia-m-gladbach",
    "monchengladbach": "borussia-m-gladbach",
    "borussia monchengladbach": "borussia-m-gladbach",
    "m gladbach": "borussia-m-gladbach",
    "cologne": "fc-cologne",
    "koln": "fc-cologne",
    "1 fc koln": "fc-cologne",
    "fc koln": "fc-cologne",
    "werder bremen": "werder-breme",
    "werder": "werder-breme",
    "augsburg": "augsbourg",
    "fc augsburg": "augsbourg",
    "heidenheim": "heidenheim",
    "1 fc heidenheim": "heidenheim",
    "st pauli": "sankt-pauli",
    "saint pauli": "sankt-pauli",
    "fc st pauli": "sankt-pauli",
    "hamburg": "hambourg-sv",
    "hamburger sv": "hambourg-sv",
    "hsv": "hambourg-sv",
    # Serie A
    "juventus": "juventus-turin",
    "juve": "juventus-turin",
    "juventus turin": "juventus-turin",
    "inter milan": "inter-milan",
    "internazionale": "inter-milan",
    "inter": "inter-milan",
    "ac milan": "ac-milan",
    "milan": "ac-milan",
    "napoli": "naples",
    "naples": "naples",
    "ssc napoli": "naples",
    "as roma": "as-rome",
    "roma": "as-rome",
    "lazio": "lazio-rome",
    "ss lazio": "lazio-rome",
    "atalanta": "atalanta-bergame",
    "atalanta bergame": "atalanta-bergame",
    "fiorentina": "fiorentina",
    "viola": "fiorentina",
    "acf fiorentina": "fiorentina",
    "torino": "torino",
    "torino fc": "torino",
    "bologna": "bologne",
    "bologna fc": "bologne",
    "genoa": "genoa",
    "genoa cfc": "genoa",
    "udinese": "udinese",
    "udinese calcio": "udinese",
    "cagliari": "cagliari",
    "cagliari calcio": "cagliari",
    "hellas verona": "hellas-verone",
    "verona": "hellas-verone",
    "como": "come",
    "como 1907": "come",
    "lecce": "lecce",
    "us lecce": "lecce",
    "parma": "parme",
    "parma calcio": "parme",
    "sassuolo": "sassuolo",
    "us sassuolo": "sassuolo",
    "cremonese": "cremonese",
    "us cremonese": "cremonese",
    "pisa": "pise",
    "ac pisa": "pise",
}

MOIS_FR = {
    # Avec accents (tels qu'affichés)
    "janv": "01", "févr": "02", "mars": "03", "avr": "04",
    "mai": "05", "juin": "06", "juil": "07", "août": "08",
    "sept": "09", "oct": "10", "nov": "11", "déc": "12",
    "janvier": "01", "février": "02", "avril": "04",
    "juillet": "07", "aout": "08", "septembre": "09",
    "octobre": "10", "novembre": "11", "décembre": "12",
    # Sans accents (après slugify_query — nécessaire car parse_date_fr normalise d'abord)
    "fevr": "02", "dec": "12",
    "fevrier": "02", "decembre": "12",
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def slugify_query(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _norm(text: str) -> str:
    """Normalise pour comparaison : supprime accents + remplace tirets par espaces."""
    return slugify_query(text).replace("-", " ")


# Nombre max de jours dans le passé pour un match (couvre toute la saison en cours ~août→mai)
MATCH_MAX_AGE_DAYS = 280

# Noms d'équipes invalides / placeholders lequipe.fr
_PLACEHOLDER_RE = re.compile(
    r"^\(?(?:qualifi[eé]|vainqueur|perdant|tbd|tba|a\s+determiner|\?|-)\)?$"
)


def _is_valid_team_name(name: str) -> bool:
    """Retourne False si le nom est un placeholder lequipe.fr comme '(qualifié)'."""
    n = _norm(name).strip()
    if len(n) < 2:
        return False
    if _PLACEHOLDER_RE.match(n):
        return False
    return True


def find_slug(team_name: str) -> str | None:
    key = slugify_query(team_name)
    return KNOWN_SLUGS.get(key)


def parse_date_fr(raw: str) -> str | None:
    """Convertit 'dimanche 19 avr. 2026' → '2026-04-19'."""
    raw_n = slugify_query(raw)
    m = re.search(r"(\d{1,2})\s+([a-z\.]{2,12})\.?\s+(\d{4})", raw_n)
    if m:
        day, month_str, year = m.group(1), m.group(2).rstrip("."), m.group(3)
        month = MOIS_FR.get(month_str)
        if month:
            return f"{year}-{month}-{int(day):02d}"
    return None


def result_for_team(team_slug: str, home_text: str, away_text: str,
                    sh: int, sa: int) -> str:
    """Retourne 'W', 'D' ou 'L' du point de vue de l'équipe."""
    home_n = _norm(home_text)
    team_n = _norm(team_slug)
    is_home = (team_n in home_n or home_n in team_n)
    gf, ga = (sh, sa) if is_home else (sa, sh)
    if gf > ga:
        return "W"
    if gf == ga:
        return "D"
    return "L"


# ---------------------------------------------------------------------------
# Scraping Playwright
# ---------------------------------------------------------------------------

def _load_playwright_page(url: str, wait_secs: int = 5):
    """Lance Playwright, charge l'URL, retourne (page, browser, playwright)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[scraper] Playwright manquant. Installez-le :")
        print("  pip install playwright && python3 -m playwright install chromium")
        sys.exit(1)

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(
        locale="fr-FR",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    page = ctx.new_page()
    try:
        page.goto(url, timeout=25000, wait_until="networkidle")
    except Exception:
        pass
    import time; time.sleep(wait_secs)

    # Accepter les cookies lequipe.fr pour accéder au contenu complet
    # Le bouton est souvent non-visible via click standard → forcer via JS
    accepted = False
    try:
        accepted = page.evaluate("""
            () => {
                const candidates = Array.from(document.querySelectorAll('button, span, a'));
                const btn = candidates.find(el => {
                    const t = el.innerText || el.textContent || '';
                    return t.trim().toLowerCase().includes("oui, j'accepte")
                        || t.trim().toLowerCase() === 'accepter';
                });
                if (btn) { btn.click(); return true; }
                return false;
            }
        """)
    except Exception:
        pass
    if accepted:
        time.sleep(2)

    # Scroll progressivement pour déclencher le lazy-loading des matchdays
    for _ in range(6):
        page.evaluate("window.scrollBy(0, 2000)")
        time.sleep(0.5)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.5)
    return page, browser, pw


def list_team_slugs(competition: str) -> list[dict]:
    """
    Retourne la liste {slug, name} des équipes d'une compétition
    en scrapant le dropdown de la page calendrier lequipe.fr.
    """
    url = f"https://www.lequipe.fr/Football/{competition}/page-calendrier-resultats"
    print(f"[scraper] Chargement du calendrier : {url}")
    page, browser, pw = _load_playwright_page(url)
    try:
        teams = page.evaluate("""
        () => {
            // Cherche les liens d'équipe dans le dropdown de la page
            const links = Array.from(document.querySelectorAll('a[href*="/page-calendrier-general/"]'));
            const seen = new Set();
            const results = [];
            for (const a of links) {
                const href = a.href;
                const slug = href.split('/page-calendrier-general/')[1]?.split('/')[0]?.split('?')[0];
                if (slug && !seen.has(slug)) {
                    seen.add(slug);
                    results.push({ slug, name: a.textContent.trim() });
                }
            }
            return results;
        }
        """)
    finally:
        browser.close()
        pw.stop()
    return teams


def scrape_team_calendar(competition: str, team_slug: str) -> list[dict]:
    """
    Scrape la page calendrier d'une équipe et retourne les matchs terminés,
    triés du plus récent au plus ancien.
    """
    url = (f"https://www.lequipe.fr/Football/{competition}"
           f"/page-calendrier-general/{team_slug}")
    print(f"[scraper] Scraping : {url}")

    page, browser, pw = _load_playwright_page(url)
    try:
        # Vérifie que la page n'est pas une 404
        body_text = page.evaluate("() => document.body.innerText")
        if ("n'existe plus" in body_text or
                "hors-jeu" in body_text or
                len(body_text) < 500):
            print(f"[scraper] Page introuvable : {url}")
            return []

        # Extrait les matchs depuis le corps de la page (texte structuré)
        matches = _parse_calendar_text(body_text, team_slug, competition)

        # Fallback DOM uniquement si le texte n'a trouvé aucun match
        if len(matches) < 1:
            matches = _parse_calendar_dom(page, team_slug, competition)

    finally:
        browser.close()
        pw.stop()

    return matches


def _parse_calendar_text(body_text: str, team_slug: str, competition: str) -> list[dict]:
    """
    Parse le texte brut de la page calendrier (innerText).

    Format lequipe.fr : une date est donnée UNE FOIS pour toute une journée,
    puis plusieurs matchs se succèdent sans répéter la date :
        vendredi 22 août 2025
        1re journée
        Bayern Munich
        6-0
        RB Leipzig
        Eintracht Francfort    ← match suivant, même date
        4-1
        Werder Brême
        ...

    Stratégie : on scanne ligne par ligne.
    - Quand on voit une date → on mémorise current_date.
    - Quand on voit un score (ex "3-1") → la ligne précédente est l'équipe domicile,
      la ligne suivante est l'équipe extérieure.
    """
    lines = [l.strip() for l in body_text.splitlines() if l.strip()]
    comp_name = COMPETITIONS.get(competition, competition)
    today = date.today()
    cutoff = today - timedelta(days=MATCH_MAX_AGE_DAYS)

    SCORE_RE = re.compile(r"^(\d{1,2})[-–](\d{1,2})$")
    # Lignes-frontières : arrêter la recherche (break) car on a dépassé le bloc de match
    SKIP_RE = re.compile(
        r"^\d{1,2}[eè]?\s*(?:journee|journée|j\.?|matchday|tour|round)"
        r"|^(?:masquer|afficher|voir|groupe|phase|poule)",
        re.IGNORECASE,
    )
    # Annotations inline dans le bloc de match (entre le nom d'équipe et le score)
    # → pas un nom d'équipe : ignorer (continue) sans stopper la recherche
    INLINE_SKIP_RE = re.compile(
        r"^(?:quarts?|demi|finale?s?|huitieme|seizieme|trente|aller|retour"
        r"|1/[248]\s*finale?|play.?off|barrages?|qualif|repechage"
        r"|[12]e\s+match|match\s+(?:aller|retour)|prolongations?|prol\.?"
        r"|t\.?a\.?b\.?(?:\s+\d+[-–]\d+)?)",
        re.IGNORECASE | re.UNICODE,
    )
    # Suffixes parasites collés aux noms d'équipes par lequipe.fr
    # (tirs au but, prolongations — ex : "Brestt.a.b. 5-4", "Lyonprol.")
    # Note : pas de \b après \. (le point est non-word, \b échouerait)
    TAB_SUFFIX_RE  = re.compile(r't\.a\.b\..*$', re.IGNORECASE)
    PROL_SUFFIX_RE = re.compile(r'prol\..*$',    re.IGNORECASE)
    TAB_LINE_RE    = re.compile(r'^t\.a\.b\.',   re.IGNORECASE)

    current_date: str | None = None
    matches: list[dict] = []

    for i, line in enumerate(lines):
        # --- Mise à jour de la date courante ---
        dv = parse_date_fr(line)
        if dv:
            try:
                current_date = dv
            except ValueError:
                pass
            continue

        # --- Détection d'un score ---
        sm = SCORE_RE.match(line)
        if sm and current_date and i > 0 and i < len(lines) - 1:
            try:
                match_date = date.fromisoformat(current_date)
            except ValueError:
                continue

            # Filtre dates
            if match_date > today or match_date < cutoff:
                continue

            sh, sa = int(sm.group(1)), int(sm.group(2))

            # Cherche l'équipe domicile : remonte jusqu'à trouver un nom valide
            # — ignore les annotations de tour CL/coupe (INLINE_SKIP_RE) → continue
            # — s'arrête sur une frontière de journée/date (SKIP_RE) → break
            home_name = ""
            home_back_idx = -1
            for back in range(i - 1, max(i - 6, -1), -1):
                candidate = lines[back]
                if SKIP_RE.match(candidate) or parse_date_fr(candidate):
                    break
                if INLINE_SKIP_RE.match(candidate):
                    continue  # annotation de round, pas un nom d'équipe
                if candidate and _is_valid_team_name(candidate):
                    home_name = candidate
                    home_back_idx = back
                    break

            # Équipe extérieure : descend jusqu'à trouver un nom valide
            away_name = ""
            away_fwd_idx = -1
            for fwd in range(i + 1, min(i + 6, len(lines))):
                candidate = lines[fwd]
                if SKIP_RE.match(candidate) or parse_date_fr(candidate):
                    break
                if INLINE_SKIP_RE.match(candidate):
                    continue  # annotation de round, pas un nom d'équipe
                if candidate and _is_valid_team_name(candidate):
                    away_name = candidate
                    away_fwd_idx = fwd
                    break

            if home_name and away_name:
                # --- Nettoyage des suffixes collés (t.a.b., prol.) ---
                home_has_tab  = bool(TAB_SUFFIX_RE.search(home_name))
                away_has_tab  = bool(TAB_SUFFIX_RE.search(away_name))
                home_has_prol = bool(PROL_SUFFIX_RE.search(home_name))
                away_has_prol = bool(PROL_SUFFIX_RE.search(away_name))

                home_name = TAB_SUFFIX_RE.sub('', home_name).strip()
                away_name = TAB_SUFFIX_RE.sub('', away_name).strip()
                home_name = PROL_SUFFIX_RE.sub('', home_name).strip()
                away_name = PROL_SUFFIX_RE.sub('', away_name).strip()

                # Vérifie qu'on a encore des noms valides après nettoyage
                if not home_name or not away_name:
                    continue

                # Détermine is_home au moment du parsing pour éviter un recalcul divergent
                home_n = _norm(home_name)
                team_n = _norm(team_slug)
                is_home = (team_n in home_n or home_n in team_n)

                # Résultat de base sur le score de temps réglementaire
                res = result_for_team(team_slug, home_name, away_name, sh, sa)

                # Correction pour les tirs au but : l'équipe dont le nom
                # portait "t.a.b." a gagné (ex. "Brestt.a.b. 5-4" → Brest gagne)
                if home_has_tab or away_has_tab:
                    team_won_tab = (is_home and home_has_tab) or (not is_home and away_has_tab)
                    res = "W" if team_won_tab else "L"

                # Format B : t.a.b. sur sa propre ligne, immédiatement après le nom
                # du vainqueur. Convention lequipe.fr :
                #   • t.a.b. juste après l'équipe extérieure → l'ext. a gagné
                #   • t.a.b. entre le nom domicile et le score → le dom. a gagné
                elif not (home_has_tab or away_has_tab):
                    standalone_tab = False
                    tab_away_wins  = False

                    # Cas 1 : t.a.b. juste après away_name
                    if (away_fwd_idx >= 0
                            and away_fwd_idx + 1 < len(lines)
                            and TAB_LINE_RE.match(lines[away_fwd_idx + 1])):
                        standalone_tab = True
                        tab_away_wins  = True

                    # Cas 2 : t.a.b. entre home_back_idx et le score
                    if not standalone_tab and home_back_idx >= 0:
                        for li in range(home_back_idx + 1, i):
                            if TAB_LINE_RE.match(lines[li]):
                                standalone_tab = True
                                tab_away_wins  = False
                                break

                    if standalone_tab:
                        if tab_away_wins:
                            res = "W" if not is_home else "L"
                        else:
                            res = "W" if is_home else "L"

                matches.append({
                    "date": current_date,
                    "competition": comp_name,
                    "home_team": home_name,
                    "away_team": away_name,
                    "score_home": sh,
                    "score_away": sa,
                    "result": res,
                    "is_home": is_home,
                })

    matches.sort(key=lambda m: m["date"], reverse=True)
    return matches


def _parse_calendar_dom(page, team_slug: str, competition: str) -> list[dict]:
    """Fallback : parse les éléments DOM .TeamScore."""
    comp_name = COMPETITIONS.get(competition, competition)
    today_str = date.today().isoformat()

    raw = page.evaluate("""
    () => {
        const results = [];
        const topScores = Array.from(document.querySelectorAll('.TeamScore')).filter(el => {
            // Ne garde que les divs .TeamScore de niveau racine (pas enfants d'un autre .TeamScore)
            return !el.parentElement?.closest('.TeamScore');
        });
        for (const el of topScores) {
            const scoreEl = el.querySelector('.TeamScore__score, .TeamScore__score--ended');
            if (!scoreEl) continue;
            const scoreText = scoreEl.textContent.trim();
            const sm = scoreText.match(/(\\d+)[-–](\\d+)/);
            if (!sm) continue;

            // Trouve les équipes (premier et dernier bloc de texte significatif)
            const allText = el.textContent.replace(scoreText, '|SCORE|').trim();
            const parts = allText.split('|SCORE|');
            const home = parts[0]?.trim().replace(/\\n/g, ' ').trim() || '';
            const away = parts[1]?.trim().replace(/\\n/g, ' ').trim() || '';

            results.push({
                scoreHome: parseInt(sm[1]),
                scoreAway: parseInt(sm[2]),
                home: home.split('\\n').pop()?.trim() || home,
                away: away.split('\\n').shift()?.trim() || away,
                isEnded: el.querySelector('.TeamScore__score--ended') !== null
            });
        }
        return results;
    }
    """)

    matches = []
    for m in raw:
        if not m.get("isEnded"):
            continue
        home = m.get("home", "")
        away = m.get("away", "")
        sh   = m.get("scoreHome", 0)
        sa   = m.get("scoreAway", 0)
        res  = result_for_team(team_slug, home, away, sh, sa)
        matches.append({
            "date": today_str,     # date inconnue dans ce fallback
            "competition": comp_name,
            "home_team": home,
            "away_team": away,
            "score_home": sh,
            "score_away": sa,
            "result": res,
        })
    return matches


# ---------------------------------------------------------------------------
# Découverte dynamique du slug via la page club lequipe.fr
# ---------------------------------------------------------------------------

def _build_team_page_urls(team_name: str) -> list[str]:
    """
    Génère plusieurs variantes d'URL de page club lequipe.fr à tester.
    Ex: "Burnley" → ["/Football/Burnley/"]
        "Manchester United" → ["/Football/Manchester-united/", "/Football/Manchester-United/"]
    """
    base = "https://www.lequipe.fr/Football/"
    words = team_name.strip().split()

    variants = []
    # Capitalise 1er mot seulement
    v1 = "-".join([words[0].capitalize()] + [w.lower() for w in words[1:]])
    # Capitalise chaque mot
    v2 = "-".join(w.capitalize() for w in words)
    # Tout en minuscule
    v3 = "-".join(w.lower() for w in words)

    for v in dict.fromkeys([v1, v2, v3]):   # dédoublonne tout en gardant l'ordre
        variants.append(f"{base}{v}/")
    return variants


def discover_team_slug(team_name: str) -> tuple[str, str] | None:
    """
    Visite la page club lequipe.fr d'une équipe inconnue et en extrait
    le slug + la compétition depuis le lien calendrier.

    Retourne (slug, competition) ou None si introuvable.
    """
    import time

    urls = _build_team_page_urls(team_name)

    for url in urls:
        page, browser, pw = _load_playwright_page(url, wait_secs=2)
        try:
            hrefs = page.evaluate(
                "() => Array.from(document.querySelectorAll('a')).map(a => a.href)"
            )
            for href in hrefs:
                m = re.search(
                    r"/Football/([^/]+)/page-calendrier-general/([^/?#]+)", href
                )
                if m:
                    competition = m.group(1)
                    slug = m.group(2)
                    print(f"[scraper] Équipe découverte : slug='{slug}' compétition='{competition}'")
                    return slug, competition
        finally:
            browser.close()
            pw.stop()

    return None


def find_team_slug_on_site(team_name: str, competition: str) -> str | None:
    """
    Cherche dynamiquement le slug d'une équipe dans le dropdown de la page
    calendrier d'une compétition donnée.
    """
    teams = list_team_slugs(competition)
    q = _norm(team_name)
    for t in teams:
        t_norm = _norm(t["name"])
        if q in t_norm or t_norm in q:
            return t["slug"]
    return None


# ---------------------------------------------------------------------------
# Filtrage des matchs par équipe
# ---------------------------------------------------------------------------

def _filter_team_matches(matches: list[dict], team_slug: str) -> list[dict]:
    """
    Ne conserve que les matchs où l'équipe (team_slug) participe réellement.
    Évite de retourner des matchs d'autres équipes si la page n'est pas la bonne.
    Normalise tirets → espaces pour gérer "real-madrid" vs "real madrid".
    """
    slug_n = _norm(team_slug)
    filtered = []
    for m in matches:
        home_n = _norm(m.get("home_team", ""))
        away_n = _norm(m.get("away_team", ""))
        if slug_n in home_n or home_n in slug_n or slug_n in away_n or away_n in slug_n:
            filtered.append(m)
    return filtered


# ---------------------------------------------------------------------------
# Cache fichier — évite de re-scraper des données déjà fraîches
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    d = Path(__file__).parent / "cache"
    d.mkdir(exist_ok=True)
    return d


def _load_cache(slug: str) -> dict | None:
    """Retourne les données en cache si elles ont moins de CACHE_TTL_HOURS, sinon None."""
    path = _cache_dir() / f"team_{slug}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        last_updated = datetime.fromisoformat(
            data.get("last_updated", "2000-01-01T00:00:00")
        )
        age_min = int((datetime.now() - last_updated).total_seconds() // 60)
        if age_min < CACHE_TTL_HOURS * 60:
            print(f"[scraper] Cache utilisé pour '{slug}' (données vieilles de {age_min} min)")
            data["from_cache"] = True
            data["cache_age_min"] = age_min
            return data
    except Exception:
        pass
    return None


def _save_cache(slug: str, data: dict) -> None:
    try:
        path = _cache_dir() / f"team_{slug}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[scraper] Cache non sauvegardé : {e}")


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

def scrape_team(team_name: str,
                competition: str | None = None,
                nb_matches: int = 5,
                force_refresh: bool = False) -> dict:
    """
    Scrape les {nb_matches} derniers matchs d'une équipe sur lequipe.fr.

    Stratégie :
     1. Résout le slug lequipe.fr depuis la table de correspondance ou dynamiquement.
     2. Vérifie le cache (retour immédiat si données fraîches et force_refresh=False).
     3. Scrape toutes les compétitions EN PARALLÈLE (championnat + CL + coupes).
     4. Combine les résultats, déduplique, garde les N plus récents.
     5. Sauvegarde le résultat dans le cache.

    Retourne un dict compatible avec app.py.
    """
    slug = find_slug(team_name)
    resolved_slug: str | None = slug
    discovered_competition: str | None = None

    # Si le slug n'est pas dans la table connue → découverte dynamique via la page club
    if not resolved_slug:
        print(f"[scraper] Slug inconnu pour '{team_name}', découverte via lequipe.fr...")
        result = discover_team_slug(team_name)
        if result:
            resolved_slug, discovered_competition = result

    # --- Vérification du cache (avant tout scraping) ---
    if resolved_slug and not force_refresh and not competition:
        cached = _load_cache(resolved_slug)
        if cached:
            return cached

    # --- Construction de la liste complète des compétitions à scraper ---
    if competition:
        all_comps = [competition]
    elif discovered_competition:
        extra = EXTRA_COMPS_BY_MAIN.get(discovered_competition, ["ligue-des-champions"])
        all_comps = list(dict.fromkeys([discovered_competition] + extra))
    else:
        main_comp = SLUG_COMPETITION.get(resolved_slug or "") if resolved_slug else None
        if main_comp:
            extra = EXTRA_COMPS_BY_MAIN.get(main_comp, ["ligue-des-champions"])
            all_comps = list(dict.fromkeys([main_comp] + extra))
        else:
            # Slug complètement inconnu : essaie toutes les ligues majeures
            all_comps = [
                "ligue-1", "championnat-d-angleterre",
                "championnat-d-espagne", "championnat-d-italie",
                "championnat-d-allemagne",
            ]

    # Résolution du slug si encore inconnu
    if not resolved_slug:
        for comp in all_comps[:1]:
            resolved_slug = find_team_slug_on_site(team_name, comp)
            if resolved_slug:
                break

    if not resolved_slug:
        raise RuntimeError(
            f"Équipe '{team_name}' introuvable sur lequipe.fr.\n\n"
            "Vérifications :\n"
            "  • Vérifiez l'orthographe du nom\n"
            "  • Précisez la compétition avec -c (ex: -c ligue-1)\n"
            f"  • Compétitions disponibles : {', '.join(COMPETITIONS.keys())}"
        )

    # Vérification du cache après résolution (cas slug découvert dynamiquement)
    if not force_refresh and not competition:
        cached = _load_cache(resolved_slug)
        if cached:
            return cached

    # --- Scraping parallèle de toutes les compétitions ---
    print(f"[scraper] Scraping parallèle ({len(all_comps)} compétitions) pour '{resolved_slug}'…")

    all_matches: list[dict] = []
    found_competitions: list[str] = []
    _slug = resolved_slug   # capture pour le thread

    def _scrape_one(comp: str) -> tuple[str, list[dict]]:
        matches = scrape_team_calendar(comp, _slug)
        return comp, _filter_team_matches(matches, _slug)

    max_workers = min(len(all_comps), 3)   # 3 browsers max en simultané
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scrape_one, comp): comp for comp in all_comps}
        for future in as_completed(futures):
            comp = futures[future]
            try:
                _, matches = future.result()
                if matches:
                    all_matches.extend(matches)
                    found_competitions.append(comp)
            except Exception as exc:
                print(f"[scraper] Erreur pour '{comp}' : {exc}")

    if not all_matches:
        raise RuntimeError(
            f"Aucun match trouvé pour '{team_name}' sur lequipe.fr.\n\n"
            "Vérifications :\n"
            f"  • Le slug essayé : '{resolved_slug}'\n"
            "  • Utilisez --list-slugs pour voir les slugs disponibles\n"
            "  • Précisez la compétition avec -c (ex: -c ligue-1)\n"
            f"  • Compétitions disponibles : {', '.join(COMPETITIONS.keys())}"
        )

    # --- Trie par date décroissante, déduplique, prend les N plus récents ---
    seen: set = set()
    unique: list[dict] = []
    for m in sorted(all_matches, key=lambda x: x["date"], reverse=True):
        key = (m["date"], m["home_team"], m["away_team"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    matches_final = unique[:nb_matches]

    # --- Calcul des statistiques ---
    wins = draws = losses = 0
    goals_for = goals_against = 0

    slug_norm = _norm(resolved_slug)
    for m in matches_final:
        res = m["result"]
        sh, sa = m["score_home"], m["score_away"]

        if "is_home" in m:
            is_home = m["is_home"]
        else:
            home_norm = _norm(m["home_team"])
            is_home = (slug_norm in home_norm or home_norm in slug_norm)

        if res == "W":
            wins += 1
        elif res == "D":
            draws += 1
        else:
            losses += 1

        if is_home:
            goals_for += sh
            goals_against += sa
        else:
            goals_for += sa
            goals_against += sh

    n = len(matches_final)
    gsa = round(goals_for / n, 2) if n else 0.0
    gca = round(goals_against / n, 2) if n else 0.0

    # Nom affiché : depuis le premier match
    first = matches_final[0]
    h_norm = _norm(first["home_team"])
    display_name = (
        first["home_team"]
        if slug_norm in h_norm or h_norm in slug_norm
        else first["away_team"]
    )

    result = {
        "team_name":          display_name,
        "team_slug":          resolved_slug,
        "source":             "lequipe.fr",
        "competitions":       found_competitions,
        "last_updated":       datetime.now().isoformat(timespec="seconds"),
        "from_cache":         False,
        "nb_matches":         n,
        "wins":               wins,
        "draws":              draws,
        "losses":             losses,
        "goals_scored_avg":   gsa,
        "goals_conceded_avg": gca,
        # Alias directs pour app.py
        "goals_scored":       gsa,
        "goals_conceded":     gca,
        "matches":            matches_final,
    }

    # --- Sauvegarde dans le cache ---
    if not competition:
        _save_cache(resolved_slug, result)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape les 5 derniers matchs d'une équipe sur lequipe.fr"
    )
    parser.add_argument(
        "team", nargs="?",
        help='Nom de l\'équipe (ex: "PSG", "Arsenal", "Bayern")'
    )
    parser.add_argument(
        "--competition", "-c",
        help=(
            "Compétition lequipe.fr (ex: ligue-1, ligue-des-champions, "
            "championnat-d-angleterre). "
            "Détection automatique si absent."
        ),
        default=None,
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier JSON de sortie (défaut: team_stats_{slug}.json)",
        default=None,
    )
    parser.add_argument(
        "--list-slugs", action="store_true",
        help="Affiche les slugs disponibles pour une compétition (utiliser avec -c)"
    )
    args = parser.parse_args()

    # --- Mode liste des slugs
    if args.list_slugs:
        comp = args.competition or "ligue-1"
        comp_name = COMPETITIONS.get(comp, comp)
        print(f"\nÉquipes disponibles — {comp_name} ({comp}) :")
        for t in list_team_slugs(comp):
            print(f"  slug: '{t['slug']}'  —  nom: {t['name']}")
        return

    if not args.team:
        parser.print_help()
        sys.exit(1)

    try:
        data = scrape_team(args.team, args.competition)
    except RuntimeError as e:
        print(f"\n[ERREUR] {e}", file=sys.stderr)
        sys.exit(1)

    slug = data["team_slug"]
    output_path = args.output or f"team_stats_{slug}.json"
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Affichage terminal
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; Z = "\033[0m"
    icons = {"W": f"{G}✓{Z}", "D": f"{Y}={Z}", "L": f"{R}✗{Z}"}

    print(f"\n{'='*58}")
    print(f"  {data['team_name']}  [lequipe.fr]")
    print(f"{'='*58}")
    print(f"  V:{data['wins']}  N:{data['draws']}  D:{data['losses']}   "
          f"sur {data['nb_matches']} matchs")
    print(f"  Buts marqués/match   : {data['goals_scored_avg']}")
    print(f"  Buts encaissés/match : {data['goals_conceded_avg']}")
    print(f"\n  {data['nb_matches']} derniers matchs :")
    for m in data["matches"]:
        icon = icons.get(m["result"], "?")
        print(f"    {icon} {m['date']}  "
              f"{m['home_team']} {m['score_home']}-{m['score_away']} {m['away_team']}"
              f"  [{m['competition']}]")
    print(f"\n  → Sauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()
