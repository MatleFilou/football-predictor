"""
Récupération des classements UEFA (clubs) et FIFA (sélections nationales).

Sources :
  UEFA clubs   → API comp.uefa.com  (page UEFA : fr.uefa.com/nationalassociations/uefarankings/tenyears/)
  FIFA nations → Page inside.fifa.com scrappée avec Playwright

Utilisation :
    from scraping_rankings import get_uefa_rank, get_fifa_rank, get_rank

    rang = get_uefa_rank("Bayern Munich")   # → 1
    rang = get_uefa_rank("PSG")             # → ex. 8
    rang = get_fifa_rank("France")          # → 1
    rang = get_fifa_rank("Brésil")          # → 6
    # Retourne 0 si non trouvé
"""

import re
import unicodedata
import requests

# Cache en mémoire pour éviter de refaire les requêtes dans la même session
_UEFA_CACHE: list = []
_FIFA_CACHE:  list = []

UEFA_API_URL = (
    "https://comp.uefa.com/v2/coefficients"
    "?coefficientRange=OVERALL"
    "&coefficientType=MEN_CLUB"
    "&language=FR"
    "&page={page}"
    "&pagesize=50"
    "&seasonYear=2026"
)

FIFA_PAGE_URL = "https://inside.fifa.com/fr/fifa-world-ranking/men"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Minuscules + suppression accents + tirets → espaces."""
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    clean = "".join(c for c in nfkd if not unicodedata.combining(c))
    return clean.replace("-", " ")


# Alias pour les noms abrégés de l'API UEFA (api_name → noms alternatifs reconnus)
_UEFA_NAME_ALIASES: dict[str, list[str]] = {
    "man city":  ["manchester city", "man city"],
    "paris":     ["psg", "paris saint germain", "paris sg", "paris saint-germain"],
    "atleti":    ["atletico madrid", "atletico de madrid", "atletico"],
    "juventus":  ["juventus turin", "juve"],
    "monaco":    ["as monaco"],
    "bayer":     ["bayer leverkusen", "leverkusen"],
    "rb leipzig":["rb leipzig", "leipzig"],
    "inter":     ["inter milan", "internazionale"],
    "man united":["manchester united", "man utd"],
    "city":      [],  # trop générique, ignoré
}


def _word_boundary_match(phrase: str, text: str) -> bool:
    """Vérifie si 'phrase' apparaît comme mot(s) entier(s) dans 'text'."""
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _match_name(query: str, candidate: str) -> bool:
    """
    Correspondance souple entre un nom recherché et un nom candidat.
    Gère :
      - correspondance exacte / sous-chaîne
      - alias connus (noms abrégés de l'API UEFA) avec frontières de mots
      - préfixe des mots (3 chars) pour München↔Munich, etc.
    """
    q = _normalize(query)
    c = _normalize(candidate)

    if q == c or q in c or c in q:
        return True

    # Alias explicites — on vérifie que alias_key est un MOT ENTIER dans c ou q
    # (pas une sous-chaîne : "bayer" ne doit pas matcher "bayern")
    for alias_key, alternatives in _UEFA_NAME_ALIASES.items():
        if not alternatives:
            continue
        if _word_boundary_match(alias_key, c):
            if q in alternatives or alias_key in q:
                return True
        if _word_boundary_match(alias_key, q):
            if c in alternatives or alias_key in c:
                return True

    # Correspondance par préfixe de mots (3 chars) → gère München/Munich, etc.
    q_words = [w for w in q.split() if len(w) >= 3]
    c_words = [w for w in c.split() if len(w) >= 3]
    if q_words and c_words:
        matched = sum(
            1 for qw in q_words
            if any(qw[:3] == cw[:3] for cw in c_words)
        )
        if matched == len(q_words):
            return True

    return False


# ---------------------------------------------------------------------------
# UEFA — Classement des clubs (10 ans)
# Source : https://fr.uefa.com/nationalassociations/uefarankings/tenyears/
# API    : comp.uefa.com/v2/coefficients
# ---------------------------------------------------------------------------

def _load_uefa_rankings() -> list[dict]:
    """
    Charge le classement UEFA clubs via l'API officielle paginée.
    Retourne [{rank, name}, ...] trié par rang.
    """
    global _UEFA_CACHE
    if _UEFA_CACHE:
        return _UEFA_CACHE

    rankings = []
    page = 1

    while True:
        try:
            resp = requests.get(
                UEFA_API_URL.format(page=page),
                headers=HEADERS,
                timeout=12,
            )
            resp.raise_for_status()
            members = resp.json().get("data", {}).get("members", [])
            if not members:
                break  # plus de pages

            for item in members:
                member       = item.get("member", {})
                overall      = item.get("overallRanking", {})
                rank         = overall.get("position", 0)
                display_name = (
                    member.get("displayName")
                    or member.get("internationalName")
                    or member.get("name", "")
                )
                if rank and display_name:
                    rankings.append({"rank": rank, "name": display_name})

            page += 1
            if page > 20:   # garde-fou
                break

        except Exception:
            break

    rankings.sort(key=lambda x: x["rank"])
    _UEFA_CACHE = rankings
    return rankings


def get_uefa_rank(team_name: str) -> int:
    """
    Retourne le rang UEFA 10 ans d'un club.
    Retourne 0 si non trouvé.

    Exemples :
        get_uefa_rank("Real Madrid")    → 2
        get_uefa_rank("PSG")            → ex. 7
        get_uefa_rank("Bayern Munich")  → 1
    """
    for entry in _load_uefa_rankings():
        if _match_name(team_name, entry["name"]):
            return entry["rank"]
    return 0


# ---------------------------------------------------------------------------
# FIFA — Classement mondial masculin
# Source : https://inside.fifa.com/fr/fifa-world-ranking/men
# ---------------------------------------------------------------------------

def _parse_fifa_row(row_text: str) -> dict | None:
    """
    Extrait rang + nom depuis une ligne de tableau FIFA.

    Format d'une ligne (innerText d'un <tr>) :
        "1\n2\t\nFrance\t\n\t\n\nFrance\n\nColombie\n\nFT\n\n3\n1\t\n+3.36\t\n1877.32"
        "4\n\t\nAngleterre\t\n..."

    Le rang est le 1er entier, le nom de pays est le 1er token non-numérique
    non vide après le rang.
    """
    tokens = [t.strip() for t in re.split(r"[\n\t]", row_text) if t.strip()]
    if not tokens:
        return None

    # 1er token doit être le rang (entier pur)
    if not re.fullmatch(r"\d{1,3}", tokens[0]):
        return None
    rank = int(tokens[0])

    # Cherche le 1er token qui ressemble à un nom de pays
    # (pas un entier, pas +/-X.XX, pas "FT"/"AET"/score)
    SKIP_RE = re.compile(r"^[\d\+\-\.]+$|^FT$|^AET$|^pen$", re.IGNORECASE)
    name = ""
    for tok in tokens[1:]:
        if not SKIP_RE.match(tok) and len(tok) >= 2:
            name = tok
            break

    if name and rank:
        return {"rank": rank, "name": name}
    return None


def _load_fifa_rankings() -> list[dict]:
    """
    Charge le classement FIFA en scrapant la page avec Playwright.
    Clique sur «Montrer le classement complet» pour obtenir toutes les équipes.
    """
    global _FIFA_CACHE
    if _FIFA_CACHE:
        return _FIFA_CACHE

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[rankings] Playwright manquant. pip install playwright && python3 -m playwright install chromium")
        return []

    import time

    rankings = []
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(
        locale="fr-FR",
        user_agent=HEADERS["User-Agent"],
    )
    page = ctx.new_page()

    try:
        try:
            page.goto(FIFA_PAGE_URL, timeout=40000, wait_until="load")
        except Exception:
            pass
        time.sleep(5)

        # Cliquer sur «Montrer le classement complet» pour déplier la liste
        try:
            page.click("text=Montrer le classement complet", timeout=5000)
            time.sleep(3)
        except Exception:
            pass

        # Extraire toutes les lignes de tableau
        rows = page.evaluate(
            "() => Array.from(document.querySelectorAll('tr'))"
            ".map(r => r.innerText.trim())"
            ".filter(t => t.length > 2)"
        )

        for row in rows:
            entry = _parse_fifa_row(row)
            if entry:
                rankings.append(entry)

    finally:
        browser.close()
        pw.stop()

    # Dédoublonne et trie
    seen = set()
    unique = []
    for e in sorted(rankings, key=lambda x: x["rank"]):
        if e["name"] not in seen:
            seen.add(e["name"])
            unique.append(e)

    _FIFA_CACHE = unique
    return unique


def get_fifa_rank(team_name: str) -> int:
    """
    Retourne le rang FIFA d'une sélection nationale.
    Retourne 0 si non trouvé.

    Exemples :
        get_fifa_rank("France")     → 1
        get_fifa_rank("Espagne")    → 2
        get_fifa_rank("Brésil")     → 6
    """
    for entry in _load_fifa_rankings():
        if _match_name(team_name, entry["name"]):
            return entry["rank"]
    return 0


# ---------------------------------------------------------------------------
# Point d'entrée unifié
# ---------------------------------------------------------------------------

def get_rank(team_name: str, is_national: bool) -> int:
    """
    - is_national=False → rang UEFA clubs (10 ans)
    - is_national=True  → rang FIFA sélections
    Retourne 0 si non trouvé.
    """
    if is_national:
        return get_fifa_rank(team_name)
    return get_uefa_rank(team_name)


def clear_cache():
    """Vide le cache pour forcer un rechargement des données."""
    global _UEFA_CACHE, _FIFA_CACHE
    _UEFA_CACHE.clear()
    _FIFA_CACHE.clear()


# ---------------------------------------------------------------------------
# Test rapide en ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("uefa", "both"):
        print("\n=== Classement UEFA Clubs (10 ans) ===")
        rankings = _load_uefa_rankings()
        print(f"  {len(rankings)} clubs chargés")
        for e in rankings[:10]:
            print(f"  #{e['rank']:3d}  {e['name']}")

        tests = ["Bayern Munich", "Real Madrid", "PSG", "Paris Saint-Germain",
                 "Arsenal", "Manchester City", "Inter Milan", "Juventus"]
        print("\n  Recherches :")
        for t in tests:
            print(f"    get_uefa_rank({t!r}) = {get_uefa_rank(t)}")

    if mode in ("fifa", "both"):
        print("\n=== Classement FIFA Nations (hommes) ===")
        rankings = _load_fifa_rankings()
        print(f"  {len(rankings)} nations chargées")
        for e in rankings[:10]:
            print(f"  #{e['rank']:3d}  {e['name']}")

        tests = ["France", "Espagne", "Brésil", "Argentine", "Angleterre", "Portugal"]
        print("\n  Recherches :")
        for t in tests:
            print(f"    get_fifa_rank({t!r}) = {get_fifa_rank(t)}")
