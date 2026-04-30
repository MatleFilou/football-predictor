"""
Récupération des passes décisives du meilleur passeur d'un club sur footmercato.net.

URL : https://www.footmercato.net/club/{slug}/effectif/
Colonne cible : PD (Passes Décisives)

Usage :
    from scraper_footmercato import get_top_assists
    nb_pd, nom = get_top_assists("PSG")         # → (9, "Vitinha")
    nb_pd, nom = get_top_assists("Arsenal")     # → (7, "...")
    nb_pd, nom = get_top_assists("Inter Milan") # → (0, "") si absent de footmercato
"""

import re
import unicodedata
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Slugs connus (vérifiés sur footmercato.net)
# ---------------------------------------------------------------------------
KNOWN_SLUGS: dict[str, str] = {
    # ── Ligue 1 ──────────────────────────────────────────────────────────────
    "psg":                          "psg",
    "paris sg":                     "psg",
    "paris saint germain":          "psg",
    "paris saint-germain":          "psg",
    "marseille":                    "om",
    "om":                           "om",
    "olympique de marseille":       "om",
    "lyon":                         "ol",
    "ol":                           "ol",
    "olympique lyonnais":           "ol",
    "monaco":                       "as-monaco",
    "as monaco":                    "as-monaco",
    "lille":                        "losc",
    "losc":                         "losc",
    "lens":                         "racing-club-de-lens",
    "rc lens":                      "racing-club-de-lens",
    "rennes":                       "stade-rennais-fc",
    "stade rennais":                "stade-rennais-fc",
    "strasbourg":                   "rc-strasbourg-alsace",
    "rc strasbourg":                "rc-strasbourg-alsace",
    "lorient":                      "fc-lorient",
    "paris fc":                     "paris-fc",
    "toulouse":                     "tfc",
    "tfc":                          "tfc",
    "brest":                        "stade-brestois-29",
    "stade brestois":               "stade-brestois-29",
    "angers":                       "angers-sco",
    "angers sco":                   "angers-sco",
    "le havre":                     "hac",
    "hac":                          "hac",
    "nice":                         "ogc-nice",
    "ogc nice":                     "ogc-nice",
    "auxerre":                      "association-jeunesse-auxerroise",
    "aja":                          "association-jeunesse-auxerroise",
    "nantes":                       "fc-nantes",
    "fc nantes":                    "fc-nantes",
    "metz":                         "fc-metz",
    "fc metz":                      "fc-metz",
    # ── La Liga ──────────────────────────────────────────────────────────────
    "real madrid":                  "real-madrid",
    "barcelona":                    "fc-barcelone",
    "fc barcelone":                 "fc-barcelone",
    "fc barcelona":                 "fc-barcelone",
    "barca":                        "fc-barcelone",
    "atletico madrid":              "atletico-madrid",
    "atletico de madrid":           "atletico-madrid",
    "atletico":                     "atletico-madrid",
    "villarreal":                   "villarreal-cf",
    "betis":                        "real-betis-balompie",
    "real betis":                   "real-betis-balompie",
    "getafe":                       "getafe-cf",
    "celta vigo":                   "real-club-celta-de-vigo",
    "celta":                        "real-club-celta-de-vigo",
    "real sociedad":                "real-sociedad",
    "athletic bilbao":              "athletic-club-bilbao",
    "bilbao":                       "athletic-club-bilbao",
    "osasuna":                      "ca-osasuna",
    "girona":                       "girona-fc",
    "espanyol":                     "reial-club-deportiu-espanyol",
    "rcd espanyol":                 "reial-club-deportiu-espanyol",
    "valence":                      "valencia-cf",
    "valencia":                     "valencia-cf",
    "majorque":                     "real-club-deportivo-majorque",
    "mallorca":                     "real-club-deportivo-majorque",
    "elche":                        "elche-cf",
    "rayo vallecano":               "rayo-vallecano",
    "rayo":                         "rayo-vallecano",
    "sevilla":                      "sevilla-fc",
    "sevilla fc":                   "sevilla-fc",
    "alaves":                       "deportivo-alaves",
    "deportivo alaves":             "deportivo-alaves",
    "levante":                      "levante-ud",
    "oviedo":                       "real-oviedo",
    "real oviedo":                  "real-oviedo",
    # ── Bundesliga ───────────────────────────────────────────────────────────
    "bayern munich":                "bayern-munich",
    "bayern":                       "bayern-munich",
    "borussia dortmund":            "borussia-dortmund",
    "dortmund":                     "borussia-dortmund",
    "bvb":                          "borussia-dortmund",
    "rb leipzig":                   "rb-leipzig",
    "leipzig":                      "rb-leipzig",
    "stuttgart":                    "vfb-stuttgart-1893",
    "vfb stuttgart":                "vfb-stuttgart-1893",
    "hoffenheim":                   "tsg-1899-hoffenheim",
    "tsg hoffenheim":               "tsg-1899-hoffenheim",
    "leverkusen":                   "bayer-04-leverkusen",
    "bayer leverkusen":             "bayer-04-leverkusen",
    "freiburg":                     "sc-fribourg",
    "fribourg":                     "sc-fribourg",
    "sc freiburg":                  "sc-fribourg",
    "eintracht frankfurt":          "eintracht-francfort",
    "frankfurt":                    "eintracht-francfort",
    "francfort":                    "eintracht-francfort",
    "augsburg":                     "fc-augsburg",
    "augsbourg":                    "fc-augsburg",
    "mainz":                        "1-fsv-mainz-05",
    "mayence":                      "1-fsv-mainz-05",
    "union berlin":                 "1-fc-union-berlin",
    "fc union berlin":              "1-fc-union-berlin",
    "cologne":                      "1-fc-cologne",
    "koln":                         "1-fc-cologne",
    "monchengladbach":              "borussia-vfl-monchengladbach",
    "gladbach":                     "borussia-vfl-monchengladbach",
    "borussia monchengladbach":     "borussia-vfl-monchengladbach",
    "hamburger sv":                 "hambourg-sv",
    "hambourg":                     "hambourg-sv",
    "hamburg":                      "hambourg-sv",
    "werder bremen":                "sv-werder-breme",
    "breme":                        "sv-werder-breme",
    "bremen":                       "sv-werder-breme",
    "st pauli":                     "fc-st-pauli",
    "fc st pauli":                  "fc-st-pauli",
    "wolfsburg":                    "vfl-wolfsbourg",
    "wolfsbourg":                   "vfl-wolfsbourg",
    "heidenheim":                   "1-fc-heidenheim-1846",
    "fc heidenheim":                "1-fc-heidenheim-1846",
    # ── Serie A ──────────────────────────────────────────────────────────────
    "inter milan":                  "inter",
    "inter":                        "inter",
    "internazionale":               "inter",
    "fc internazionale":            "inter",
    "ac milan":                     "ac-milan",
    "milan":                        "ac-milan",
    "napoli":                       "naples",
    "naples":                       "naples",
    "ssc napoli":                   "naples",
    "juventus":                     "juventus",
    "juve":                         "juventus",
    "come":                         "calcio-come",
    "como":                         "calcio-come",
    "roma":                         "as-rome",
    "as roma":                      "as-rome",
    "atalanta":                     "atalanta-bergame",
    "atalanta bergame":             "atalanta-bergame",
    "bologna":                      "bologne-fc-1909",
    "bologne":                      "bologne-fc-1909",
    "lazio":                        "ss-lazio",
    "ss lazio":                     "ss-lazio",
    "sassuolo":                     "us-sassuolo-calcio",
    "udinese":                      "udinese-calcio",
    "torino":                       "torino-fc",
    "genoa":                        "genoa-cfc",
    "parma":                        "parme-fc",
    "parme":                        "parme-fc",
    "fiorentina":                   "acf-fiorentina",
    "acf fiorentina":               "acf-fiorentina",
    "cagliari":                     "cagliari-calcio",
    "cremonese":                    "us-cremonese",
    "lecce":                        "us-lecce",
    "hellas verona":                "hellas-verona-fc",
    "verona":                       "hellas-verona-fc",
    "pisa":                         "ac-pisa-1909",
    # ── Premier League ───────────────────────────────────────────────────────
    "manchester city":              "manchester-city",
    "man city":                     "manchester-city",
    "arsenal":                      "arsenal",
    "liverpool":                    "liverpool",
    "chelsea":                      "chelsea",
    "tottenham":                    "tottenham-hotspur",
    "spurs":                        "tottenham-hotspur",
    "tottenham hotspur":            "tottenham-hotspur",
    "manchester united":            "manchester-united",
    "man united":                   "manchester-united",
    "man utd":                      "manchester-united",
    "aston villa":                  "aston-villa",
    "newcastle":                    "newcastle",
    "newcastle united":             "newcastle",
    "brighton":                     "brighton-hove-albion-fc",
    "brighton hove albion":         "brighton-hove-albion-fc",
    "bournemouth":                  "bournemouth",
    "brentford":                    "brentford-fc",
    "everton":                      "everton",
    "sunderland":                   "sunderland-afc",
    "fulham":                       "fulham-fc",
    "crystal palace":               "crystal-palace-fc",
    "leeds":                        "leeds-united-fc",
    "leeds united":                 "leeds-united-fc",
    "nottingham forest":            "nottingham-forest-fc",
    "nottingham":                   "nottingham-forest-fc",
    "west ham":                     "west-ham-united",
    "west ham united":              "west-ham-united",
    "burnley":                      "burnley-fc",
    "wolverhampton":                "wolverhampton-wanderers",
    "wolves":                       "wolverhampton-wanderers",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

# Pages classement passeurs par ligue (stats championnat uniquement)
LEAGUE_PASSEUR_URLS: dict[str, str] = {
    "ligue-1":          "https://www.footmercato.net/france/ligue-1/passeur",
    "la-liga":          "https://www.footmercato.net/espagne/liga/passeur",
    "bundesliga":       "https://www.footmercato.net/allemagne/bundesliga/passeur",
    "serie-a":          "https://www.footmercato.net/italie/serie-a/passeur",
    "premier-league":   "https://www.footmercato.net/angleterre/premier-league/passeur",
}

# Slug club → ligue
SLUG_TO_LEAGUE: dict[str, str] = {
    # Ligue 1
    "psg": "ligue-1", "om": "ligue-1", "ol": "ligue-1", "as-monaco": "ligue-1",
    "losc": "ligue-1", "ogc-nice": "ligue-1", "racing-club-de-lens": "ligue-1",
    "stade-rennais-fc": "ligue-1", "rc-strasbourg-alsace": "ligue-1",
    "fc-lorient": "ligue-1", "paris-fc": "ligue-1", "tfc": "ligue-1",
    "stade-brestois-29": "ligue-1", "angers-sco": "ligue-1", "hac": "ligue-1",
    "association-jeunesse-auxerroise": "ligue-1", "fc-nantes": "ligue-1",
    "fc-metz": "ligue-1",
    # La Liga
    "real-madrid": "la-liga", "fc-barcelone": "la-liga", "atletico-madrid": "la-liga",
    "villarreal-cf": "la-liga", "real-betis-balompie": "la-liga", "getafe-cf": "la-liga",
    "real-club-celta-de-vigo": "la-liga", "real-sociedad": "la-liga",
    "athletic-club-bilbao": "la-liga", "ca-osasuna": "la-liga", "girona-fc": "la-liga",
    "reial-club-deportiu-espanyol": "la-liga", "valencia-cf": "la-liga",
    "real-club-deportivo-majorque": "la-liga", "elche-cf": "la-liga",
    "rayo-vallecano": "la-liga", "sevilla-fc": "la-liga", "deportivo-alaves": "la-liga",
    "levante-ud": "la-liga", "real-oviedo": "la-liga",
    # Bundesliga
    "bayern-munich": "bundesliga", "borussia-dortmund": "bundesliga",
    "rb-leipzig": "bundesliga", "vfb-stuttgart-1893": "bundesliga",
    "tsg-1899-hoffenheim": "bundesliga", "bayer-04-leverkusen": "bundesliga",
    "sc-fribourg": "bundesliga", "eintracht-francfort": "bundesliga",
    "fc-augsburg": "bundesliga", "1-fsv-mainz-05": "bundesliga",
    "1-fc-union-berlin": "bundesliga", "1-fc-cologne": "bundesliga",
    "borussia-vfl-monchengladbach": "bundesliga", "hambourg-sv": "bundesliga",
    "sv-werder-breme": "bundesliga", "fc-st-pauli": "bundesliga",
    "vfl-wolfsbourg": "bundesliga", "1-fc-heidenheim-1846": "bundesliga",
    # Serie A
    "inter": "serie-a", "ac-milan": "serie-a", "naples": "serie-a",
    "juventus": "serie-a", "calcio-come": "serie-a", "as-rome": "serie-a",
    "atalanta-bergame": "serie-a", "bologne-fc-1909": "serie-a", "ss-lazio": "serie-a",
    "us-sassuolo-calcio": "serie-a", "udinese-calcio": "serie-a", "torino-fc": "serie-a",
    "genoa-cfc": "serie-a", "parme-fc": "serie-a", "acf-fiorentina": "serie-a",
    "cagliari-calcio": "serie-a", "us-cremonese": "serie-a", "us-lecce": "serie-a",
    "hellas-verona-fc": "serie-a", "ac-pisa-1909": "serie-a",
    # Premier League
    "manchester-city": "premier-league", "arsenal": "premier-league",
    "manchester-united": "premier-league", "aston-villa": "premier-league",
    "liverpool": "premier-league", "brighton-hove-albion-fc": "premier-league",
    "bournemouth": "premier-league", "chelsea": "premier-league",
    "brentford-fc": "premier-league", "everton": "premier-league",
    "sunderland-afc": "premier-league", "fulham-fc": "premier-league",
    "crystal-palace-fc": "premier-league", "newcastle": "premier-league",
    "leeds-united-fc": "premier-league", "nottingham-forest-fc": "premier-league",
    "west-ham-united": "premier-league", "tottenham-hotspur": "premier-league",
    "burnley-fc": "premier-league", "wolverhampton-wanderers": "premier-league",
}

# Cache des pages passeurs (ligue → liste de joueurs)
_LEAGUE_CACHE: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    clean = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", clean).strip()


def _find_slug(team_name: str) -> str | None:
    """Retourne le slug footmercato pour un nom d'équipe, ou None."""
    norm = _normalize(team_name)
    norm_tokens = set(norm.split())
    if norm in KNOWN_SLUGS:
        return KNOWN_SLUGS[norm]
    for key, slug in KNOWN_SLUGS.items():
        if set(key.split()) == norm_tokens:
            return slug
    for key, slug in KNOWN_SLUGS.items():
        key_tokens = set(key.split())
        if len(key_tokens) >= 2 and key_tokens.issubset(norm_tokens):
            return slug
    return None


# Mots-clés des ligues dans les pages joueur (pour identifier la ligne championnat)
LEAGUE_KEYWORDS: dict[str, list[str]] = {
    "ligue-1":        ["ligue 1"],
    "la-liga":        ["laliga", "liga", "primera"],
    "bundesliga":     ["bundesliga"],
    "serie-a":        ["serie a"],
    "premier-league": ["premier league"],
}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _fetch_with_retry(url: str) -> requests.Response:
    import time
    for attempt in range(2):
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 429:
            time.sleep(1.5)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Rate-limited : {url}")


def _get_players_from_effectif(club_slug: str) -> list[dict]:
    """
    Scrape la page /effectif/ d'un club.
    Retourne [{name, pd_all_comp, player_url}, ...] triés par pd desc.
    pd_all_comp = passes décisives toutes compétitions confondues.
    """
    url = f"https://www.footmercato.net/club/{club_slug}/effectif/"
    r = _fetch_with_retry(url)
    soup = BeautifulSoup(r.text, "html.parser")

    players = []
    for table in soup.find_all("table"):
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        if "PD" not in ths:
            continue
        pd_idx = ths.index("PD")
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) <= pd_idx:
                continue
            link = cols[1].find("a", href=True)
            player_url = link["href"] if link else None
            raw_name = cols[1].get_text(separator=" ", strip=True)
            name = re.sub(r"\s*\d+\s*ans?\s*$", "", raw_name, flags=re.IGNORECASE).strip()
            try:
                pd_val = int(cols[pd_idx].get_text(strip=True))
            except ValueError:
                pd_val = 0
            players.append({"name": name, "pd_all": pd_val, "url": player_url})

    players.sort(key=lambda p: p["pd_all"], reverse=True)
    return players


def _get_league_pd_for_player(player_url: str, league: str) -> int:
    """
    Scrape la page individuelle d'un joueur et retourne ses PD
    dans le championnat demandé (ligne "Ligue 1", "Liga", etc.).
    Retourne -1 si non trouvé.
    """
    import time
    keywords = LEAGUE_KEYWORDS.get(league, [])
    if not keywords or not player_url:
        return -1

    try:
        r = _fetch_with_retry(player_url)
    except Exception:
        return -1

    soup = BeautifulSoup(r.text, "html.parser")

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            row_label = cols[0].get_text(strip=True).lower()
            if any(kw in row_label for kw in keywords):
                # Structure : [competition, MJ, Buts, PD, ...]
                try:
                    return int(cols[3].get_text(strip=True))
                except (ValueError, IndexError):
                    pass
    return -1


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def get_top_assists(team_name: str) -> tuple[int, str]:
    """
    Retourne (nb_passes_decisives, nom_du_joueur) pour le meilleur passeur
    du club donné en championnat (stats ligue uniquement).

    Stratégie :
      1. Scrape /effectif/ → top 5 joueurs par PD toutes compétitions
      2. Pour chacun, lit la page individuelle → extrait les PD du championnat
      3. Retourne le maximum

    Retourne (0, "") si le club n'est pas trouvé.
    """
    import time

    slug = _find_slug(team_name)
    if not slug:
        return 0, ""

    league = SLUG_TO_LEAGUE.get(slug)
    if not league:
        return 0, ""

    try:
        players = _get_players_from_effectif(slug)
    except Exception:
        return 0, ""

    # On vérifie les 5 meilleurs passeurs (toutes compétitions)
    # pour trouver celui avec le max de PD en championnat
    candidates = [p for p in players if p["pd_all"] > 0][:5]
    if not candidates:
        return 0, ""

    best_pd, best_name = 0, ""
    for p in candidates:
        time.sleep(0.4)  # évite le rate-limiting
        league_pd = _get_league_pd_for_player(p["url"], league)
        if league_pd > best_pd:
            best_pd = league_pd
            best_name = p["name"]

    if best_pd <= 0:
        return 0, ""
    return best_pd, best_name


# ---------------------------------------------------------------------------
# Test en ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    teams = sys.argv[1:] or [
        "PSG", "Marseille", "Real Madrid", "Arsenal",
        "Bayern Munich", "Atletico Madrid", "Manchester City",
        "Juventus", "AC Milan", "Inter Milan", "Napoli",
    ]
    print(f"\n{'Équipe':<25} {'Top passeur':<25} {'PD':>4}")
    print("-" * 58)
    for team in teams:
        pd, name = get_top_assists(team)
        if pd:
            print(f"{team:<25} {name:<25} {pd:>4}")
        else:
            print(f"{team:<25} {'— non disponible':<25}")
