import requests

API_KEY = "5c232073a22c44628a5814b8cdb3a433"

HEADERS = {
    "X-Auth-Token": API_KEY
}

BASE_URL = "https://api.football-data.org/v4"


def get_team_id(team_name):
    url = f"{BASE_URL}/teams"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("Erreur API /teams :", response.status_code, response.text)
        return None

    teams = response.json().get("teams", [])

    team_name_lower = team_name.strip().lower()

    for team in teams:
        api_name = team["name"].strip().lower()

        if team_name_lower in api_name or api_name in team_name_lower:
            return team["id"]

    return None


def get_last_5_matches(team_id):
    url = f"{BASE_URL}/teams/{team_id}/matches?status=FINISHED&limit=5"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("Erreur API /matches :", response.status_code, response.text)
        return None

    matches = response.json().get("matches", [])

    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    counted_matches = 0

    for match in matches:
        home_team_id = match["homeTeam"]["id"]
        away_team_id = match["awayTeam"]["id"]

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        counted_matches += 1

        if team_id == home_team_id:
            gf = home_goals
            ga = away_goals
        else:
            gf = away_goals
            ga = home_goals

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    if counted_matches == 0:
        return None

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored": round(goals_for / counted_matches, 2),
        "goals_conceded": round(goals_against / counted_matches, 2)
    }


def build_team_data(team_name, uefa_rank):
    team_id = get_team_id(team_name)

    if not team_id:
        return None

    stats = get_last_5_matches(team_id)

    if not stats:
        return None

    return {
        "wins": stats["wins"],
        "draws": stats["draws"],
        "losses": stats["losses"],
        "goals_scored": stats["goals_scored"],
        "goals_conceded": stats["goals_conceded"],
        "max_goals_by_player": 2,
        "max_assists_by_player": 2,
        "top_scorer_present": True,
        "top_assist_present": True,
        "uefa_rank": uefa_rank
    }
