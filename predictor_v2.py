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
# Impact joueurs clés (V1 inchangée)
# ---------------------------------------------------------------------------

def player_impact(max_goals, max_assists, scorer_present, passer_present):
    impact = 0
    impact += max_goals * (0.08 if scorer_present else -0.08)
    impact += max_assists * (0.04 if passer_present else -0.04)
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
# Basé sur la probabilité du résultat le plus probable.
# Un match à 3 issues équiprobables (33% chacune) donne 0.
# Un résultat à 100% de probabilité donne 100.
#
# Formule : indice = (prob_max - 1/3) / (2/3) × 100
# → ramène la plage [33%…100%] sur [0…100]
# ---------------------------------------------------------------------------

def confidence_index(res, home=None, away=None):
    prob_max = max(res["home_prob"], res["draw_prob"], res["away_prob"])
    indice = (prob_max - 1/3) / (2/3) * 100
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
    over25 = btts = 0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_probability(home_goals_avg, hg) * poisson_probability(away_goals_avg, ag)
            if hg + ag >= 3:
                over25 += p
            if hg > 0 and ag > 0:
                btts += p
    return {"over25": round(over25 * 100, 1), "btts": round(btts * 100, 1)}


# ---------------------------------------------------------------------------
# Value bet & recommandation finale — inchangés
# ---------------------------------------------------------------------------

def implied_prob(odd):
    return round(100 / odd, 1)


def value_bet(model_prob, odd):
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


def final_reco(index, edge):
    if index > 65 and edge > 5:
        return "🔥 PARI FORT"
    elif index > 55 and edge > 0:
        return "✅ BON PARI"
    elif index > 45:
        return "⚠️ PRUDENCE"
    else:
        return "❌ A EVITER"
