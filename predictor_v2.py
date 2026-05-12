import math


# ---------------------------------------------------------------------------
# Conversion rang → puissance
# ---------------------------------------------------------------------------

def uefa_rank_to_power(rank):
    if rank == 0:
        return 1
    elif rank <= 10:
        return 10
    elif rank <= 20:
        return 9
    elif rank <= 30:
        return 8
    elif rank <= 50:
        return 7
    elif rank <= 75:
        return 6
    elif rank <= 100:
        return 5
    elif rank <= 150:
        return 4
    elif rank <= 200:
        return 3
    else:
        return 2


def fifa_rank_to_power(rank):
    if rank == 0:
        return 1
    elif rank <= 10:
        return 10
    elif rank <= 20:
        return 9
    elif rank <= 30:
        return 8
    elif rank <= 50:
        return 7
    elif rank <= 75:
        return 6
    elif rank <= 100:
        return 5
    elif rank <= 150:
        return 4
    elif rank <= 200:
        return 3
    else:
        return 2


def get_rank_power(team):
    if "fifa_rank" in team:
        return fifa_rank_to_power(team["fifa_rank"])
    return uefa_rank_to_power(team["uefa_rank"])


# ---------------------------------------------------------------------------
# Validation forme
# ---------------------------------------------------------------------------

def validate_results(w, d, l):
    n = w + d + l
    if n < 1:
        raise ValueError("Il faut au moins 1 match (victoires + nuls + défaites ≥ 1).")
    if n > 10:
        raise ValueError("Maximum 10 matchs pris en compte.")


def form_score(w, d, l):
    validate_results(w, d, l)
    n = w + d + l
    return (w * 3 + d) / n


# ---------------------------------------------------------------------------
# Statut joueur → coefficient
# ---------------------------------------------------------------------------

def _status_coef(val) -> float:
    """
    Convertit le statut d'un joueur en coefficient multiplicateur.
      - True  / 'Titulaire' → 1.0  (titulaire : plein impact)
      - 'Banc'              → 0.5  (remplaçant : impact partiel)
      - False / 'Absent'   → 0.0  (absent : malus appliqué)
    Compatible avec les anciennes valeurs booléennes (sélections internationales).
    """
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    s = str(val).lower()
    if s == "titulaire":
        return 1.0
    if s == "banc":
        return 0.5
    return 0.0  # absent ou inconnu


# ---------------------------------------------------------------------------
# Impact joueurs clés
# ---------------------------------------------------------------------------

def player_impact(max_goals, max_assists, scorer_val, passer_val):
    sc = _status_coef(scorer_val)
    pa = _status_coef(passer_val)
    # Titulaire → bonus plein ; Banc → bonus proportionnel ; Absent → malus
    impact  = max_goals  * (0.08 * sc if sc > 0 else -0.08)
    impact += max_assists * (0.04 * pa if pa > 0 else -0.04)
    return impact


# ---------------------------------------------------------------------------
# NOUVEAU — Malus absences dans le 11 type
#
# Seuils définis par l'utilisateur :
#   0 absent  → 0.00  (neutre)
#   1 absent  → 0.05  (léger)
#   2-3 abs.  → 0.12  (modéré)
#   4+  abs.  → 0.22  (fort)
# ---------------------------------------------------------------------------

def absence_penalty(nb_absents: int) -> float:
    if nb_absents == 0:
        return 0.0
    elif nb_absents == 1:
        return 0.05
    elif nb_absents <= 3:
        return 0.12
    else:
        return 0.22


# ---------------------------------------------------------------------------
# NOUVEAU — Impact joueur clé (star de l'équipe)
#
# key_player_present : bool
#   → Absent   : malus -0.15
#   → Présent  : bonus +0.10
# ---------------------------------------------------------------------------

def key_player_impact(key_player_present: bool) -> float:
    return 0.10 if key_player_present else -0.15


# ---------------------------------------------------------------------------
# Force globale d'une équipe (V2)
#
# Nouveaux champs attendus dans le dict team :
#   nb_absents         : int  — nombre d'absents dans le 11 type
#   key_player_present : bool — joueur clé présent ?
#   key_player_form    : int  — forme du joueur clé (1/2/3), ignoré si absent
# ---------------------------------------------------------------------------

def team_strength(team, is_home=False):
    form    = form_score(team["wins"], team["draws"], team["losses"])
    power   = get_rank_power(team) * 0.12
    players = player_impact(
        team["max_goals_by_player"],
        team["max_assists_by_player"],
        team["top_scorer_present"],
        team["top_assist_present"],
    )
    absence  = absence_penalty(team.get("nb_absents", 0))
    key_p    = key_player_impact(team.get("key_player_present", True))

    strength = (
        form   * 0.5
        + team["goals_scored"]   * 0.25
        - team["goals_conceded"] * 0.2
        + players
        + power
        - absence
        + key_p
    )

    if is_home:
        strength += 0.3

    return strength


# ---------------------------------------------------------------------------
# Prédiction résultat
# ---------------------------------------------------------------------------

def predict_match(home, away):
    hs = team_strength(home, True)
    aw = team_strength(away)

    diff      = hs - aw
    draw      = max(0.15, 0.30 - abs(diff) * 0.05)
    remaining = 1 - draw

    home_p = remaining * (1 / (1 + math.exp(-diff)))
    away_p = remaining - home_p

    if home_p > away_p and home_p > draw:
        prediction, label = "1", "Victoire domicile"
    elif away_p > home_p and away_p > draw:
        prediction, label = "2", "Victoire extérieur"
    else:
        prediction, label = "N", "Match nul"

    return {
        "home_prob":  round(home_p * 100, 1),
        "draw_prob":  round(draw   * 100, 1),
        "away_prob":  round(away_p * 100, 1),
        "prediction": prediction,
        "label":      label,
    }


# ---------------------------------------------------------------------------
# Indice de confiance — échelle 0 à 100
#
# Composantes :
#   1. Score de base   : prob_max ramenée sur [33%…68%] → [0…100]
#      (68% est un plafond réaliste en football, rares sont les matchs
#       où une issue dépasse cette probabilité)
#   2. Bonus écart     : plus le favori écrase les autres issues, plus
#      l'indice monte (max +15 pts)
#   3. Malus données   : peu de matchs (<5) → moins fiable (-8 à -15)
#   4. Malus classement: aucun ranking disponible pour les deux équipes
#      → le modèle dispose de moins d'information (-10)
# ---------------------------------------------------------------------------

def confidence_index(res, home=None, away=None):
    probs = [res["home_prob"], res["draw_prob"], res["away_prob"]]
    prob_max  = max(probs)
    prob_sorted = sorted(probs, reverse=True)

    # 1. Score de base : [33.3%…68%] → [0…100]
    base = (prob_max - 33.33) / (68.0 - 33.33) * 100
    base = max(0.0, min(100.0, base))

    # 2. Bonus écart entre 1er et 2ème (max +15)
    gap       = prob_sorted[0] - prob_sorted[1]
    gap_bonus = min(15.0, gap * 0.9)

    # 3. Malus données insuffisantes
    data_malus = 0.0
    if home and away:
        nb_h = home["wins"] + home["draws"] + home["losses"]
        nb_a = away["wins"] + away["draws"] + away["losses"]
        if nb_h < 4 or nb_a < 4:
            data_malus += 15.0
        elif nb_h < 6 or nb_a < 6:
            data_malus += 8.0

        # 4. Malus classement absent pour les deux équipes
        rank_h = home.get("uefa_rank", home.get("fifa_rank", 0))
        rank_a = away.get("uefa_rank", away.get("fifa_rank", 0))
        if rank_h == 0 and rank_a == 0:
            data_malus += 10.0

    indice = base + gap_bonus - data_malus
    return round(max(0, min(100, indice)))


# ---------------------------------------------------------------------------
# Marchés de buts (Poisson) — inchangé
# ---------------------------------------------------------------------------

def adjust_expected_goals(goals_scored, max_goals, max_assists,
                           scorer_present, passer_present, rank_power):
    adjusted = goals_scored
    adjusted += max_goals  * (0.06 if scorer_present  else -0.06)
    adjusted += max_assists * (0.03 if passer_present else -0.03)
    adjusted += rank_power  * 0.04
    if adjusted < 0.2:
        adjusted = 0.2
    return round(adjusted, 2)


def poisson_probability(lmbda, k):
    return (lmbda ** k * math.exp(-lmbda)) / math.factorial(k)


def predict_scores(home_goals_avg, away_goals_avg, max_goals=10):
    scores = []
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_probability(home_goals_avg, hg) * poisson_probability(away_goals_avg, ag)
            scores.append({"home_goals": hg, "away_goals": ag, "probability": p})
    scores.sort(key=lambda x: x["probability"], reverse=True)
    return scores[0], scores[:3]


def calculate_goal_markets(home_goals_avg, away_goals_avg, max_goals=10):
    over15 = over25 = btts = 0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_probability(home_goals_avg, hg) * poisson_probability(away_goals_avg, ag)
            if hg + ag >= 2:
                over15 += p
            if hg + ag >= 3:
                over25 += p
            if hg > 0 and ag > 0:
                btts += p
    return {
        "over15": round(over15 * 100, 1),
        "over25": round(over25 * 100, 1),
        "btts":   round(btts   * 100, 1),
    }


# ---------------------------------------------------------------------------
# Value bet & recommandation finale — inchangés
# ---------------------------------------------------------------------------

def implied_prob(odd):
    if not odd or odd <= 1.0:
        return 0.0
    return round(100 / odd, 1)


def value_bet(model_prob, odd):
    if not odd or odd <= 1.0:
        return "—", 0.0, 0.0
    book  = implied_prob(odd)
    edge  = round(model_prob - book, 1)
    if edge > 10:
        label = "🔥 GROS VALUE"
    elif edge > 5:
        label = "✅ VALUE"
    elif edge > 0:
        label = "👍 Léger value"
    else:
        label = "❌ Pas intéressant"
    return label, book, edge


def final_reco(index, edge=None):
    if index >= 65:
        return "🔥 FORTE CONFIANCE"
    elif index >= 50:
        return "✅ BONNE CONFIANCE"
    elif index >= 35:
        return "⚠️ CONFIANCE MODÉRÉE"
    else:
        return "❌ MATCH INCERTAIN"


def compute_safety_score(market_code: str, prob: float, home: dict, away: dict) -> float:
    """
    Score de sécurité ajusté : combine la probabilité du modèle avec les
    indicateurs de qualité des équipes (forme, buts marqués/encaissés,
    présence des joueurs clés, buts du meilleur buteur de la saison).
    Plus le score est élevé, plus le pari est considéré sûr.
    """
    def _team_factor(team: dict) -> float:
        n = team["wins"] + team["draws"] + team["losses"]
        form     = (team["wins"] * 3 + team["draws"]) / (n * 3) if n > 0 else 0.33
        scoring  = min(team.get("goals_scored", 1.0) / 3.0, 1.0)
        defense  = max(0.0, 1.0 - team.get("goals_conceded", 1.5) / 3.0)
        key_p    = (
            _status_coef(team.get("top_scorer_present", True)) * 0.5
            + _status_coef(team.get("top_assist_present", True)) * 0.5
        )
        season_goals   = team.get("season_goals_scorer", 0)
        season_status  = team.get("season_scorer_present", True)
        scorer_bonus   = min(season_goals / 20.0, 1.0) * _status_coef(season_status)
        return (
            form          * 0.40
            + scoring     * 0.20
            + defense     * 0.20
            + key_p       * 0.10
            + scorer_bonus * 0.10
        )

    def _offensive_factor(team: dict) -> float:
        """
        Capacité offensive fiable : buts marqués normalisés × fiabilité des
        buteurs (buteur 5 matchs + buteur de la saison).
        Utilisé pour évaluer la sécurité des marchés de buts (BTTS, Over 2.5).
        """
        goals_avg    = team.get("goals_scored", 1.0)
        scorer_coef  = _status_coef(team.get("top_scorer_present", True))
        season_goals = team.get("season_goals_scorer", 0)
        season_coef  = _status_coef(team.get("season_scorer_present", True))
        goals_norm   = min(goals_avg / 2.5, 1.0)
        # Modulateur : buteurs titulaires = fiabilité maximale, absents = réduction
        scorer_factor = 0.60 + 0.25 * scorer_coef + 0.15 * min(season_goals / 15.0, 1.0) * season_coef
        return goals_norm * scorer_factor

    hs  = _team_factor(home)
    aws = _team_factor(away)
    avg = (hs + aws) / 2.0

    off_h = _offensive_factor(home)
    off_a = _offensive_factor(away)

    factors = {
        "1":        hs,
        "2":        aws,
        "N":        avg * 0.8,
        "1N":       hs * 0.65 + avg * 0.35,
        "12":       max(hs, aws),
        "2N":       aws * 0.65 + avg * 0.35,
        # Over 1.5 : seuil bas → capacité offensive globale (légèrement favorisé vs 2.5)
        "Over 1.5": (off_h + off_a) / 2.0 * 1.05,
        # Over 2.5 : total de buts → moyenne des capacités offensives des 2 équipes
        "Over 2.5": (off_h + off_a) / 2.0,
        # BTTS : les DEUX équipes doivent marquer → facteur limitant (min)
        "BTTS":     min(off_h, off_a),
    }
    factor = factors.get(market_code, avg)
    return round(prob * (0.65 + 0.35 * factor), 2)


def value_signal(edge):
    """Signal value séparé, uniquement affiché quand les cotes sont renseignées."""
    if edge > 10:
        return "🔥 GROS VALUE"
    elif edge > 5:
        return "✅ VALUE"
    elif edge > 0:
        return "👍 Léger value"
    else:
        return "❌ Pas de value"
