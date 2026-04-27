import csv
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from api_football import search_team, fetch_team_data
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False

from predictor_v2 import (
    adjust_expected_goals,
    calculate_goal_markets,
    confidence_index,
    final_reco,
    get_rank_power,
    predict_match,
    predict_scores,
    value_bet,
)

st.set_page_config(page_title="Mode Parieur", layout="wide")


HISTORIQUE_PATH = "historique.csv"
HISTORIQUE_COLUMNS = [
    "date", "competition",
    "equipe_domicile", "equipe_exterieur",
    "wins_h", "draws_h", "losses_h", "goals_scored_h", "goals_conceded_h",
    "max_goals_player_h", "max_assists_player_h",
    "top_scorer_present_h", "top_assist_present_h", "rang_h",
    "wins_a", "draws_a", "losses_a", "goals_scored_a", "goals_conceded_a",
    "max_goals_player_a", "max_assists_player_a",
    "top_scorer_present_a", "top_assist_present_a", "rang_a",
    "cote_1", "cote_n", "cote_2", "cote_over25", "cote_btts",
    "prob_domicile", "prob_nul", "prob_exterieur",
    "prediction", "indice_confiance",
    "score_probable", "over25_pct", "btts_pct",
    "value_1", "edge_1", "value_n", "edge_n", "value_2", "edge_2",
    "value_over25", "edge_over25", "value_btts", "edge_btts",
    "meilleur_marche", "verdict",
    "pari_realise", "resultat_reel", "score_reel",
]

def init_historique():
    if not os.path.exists(HISTORIQUE_PATH):
        with open(HISTORIQUE_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HISTORIQUE_COLUMNS).writeheader()

def log_analyse(row: dict):
    init_historique()
    with open(HISTORIQUE_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HISTORIQUE_COLUMNS, extrasaction="ignore").writerow(row)

def format_market_label(code):
    return {
        "1":    "1 — Victoire domicile",
        "N":    "N — Match nul",
        "2":    "2 — Victoire extérieur",
        "1N":   "1N — Domicile ou nul",
        "12":   "12 — Domicile ou extérieur",
        "2N":   "2N — Extérieur ou nul",
        "Over 2.5": "Over 2.5",
        "BTTS": "BTTS — Les 2 équipes marquent",
    }.get(code, code)

def render_lequipe_search(key_prefix: str, is_international: bool = False):
    """
    Charge les stats (V/N/D, buts) depuis lequipe.fr
    et le classement UEFA (clubs) ou FIFA (sélections) automatiquement.
    """
    col_name, col_btn, col_refresh = st.columns([3, 1, 1])
    with col_name:
        placeholder = (
            "Ex: France, Espagne, Argentine, Brésil..."
            if is_international
            else "Ex: PSG, Arsenal, Real Madrid, Bayern, Marseille..."
        )
        team_name = st.text_input(
            "Nom de l'équipe",
            key=f"{key_prefix}_lequipe_name",
            placeholder=placeholder,
        )
    with col_btn:
        st.write("")
        load_clicked = st.button("📥 Charger", key=f"{key_prefix}_lequipe_btn")
    with col_refresh:
        st.write("")
        force_refresh = st.checkbox("🔄 MAJ forcée", key=f"{key_prefix}_force_refresh",
                                    help="Ignore le cache et re-scrape les données en ligne")

    if load_clicked and team_name.strip():
        name = team_name.strip()

        # --- Classement (FIFA ou UEFA selon l'onglet) ---
        rank_msg = ""
        try:
            from scraping_rankings import get_rank
            rank = get_rank(name, is_national=is_international)
            if rank:
                st.session_state[f"{key_prefix}_rank"] = rank
                rank_msg = f"  Classement {'FIFA' if is_international else 'UEFA'} : **#{rank}**."
        except Exception:
            pass

        if is_international:
            # Pour les sélections : pas de scraping lequipe.fr (clubs seulement)
            if rank_msg:
                st.success(f"✅ Classement chargé pour **{name}**.{rank_msg}\nComplétez la forme manuellement.")
            else:
                st.warning(f"Sélection **{name}** non trouvée dans le classement FIFA. Entrez le rang manuellement.")
            return

        # --- Stats lequipe.fr (clubs uniquement) ---
        spinner_msg = (
            f"Récupération des stats pour {name} sur lequipe.fr…"
            if force_refresh
            else f"Chargement des stats pour {name}…"
        )
        with st.spinner(spinner_msg):
            try:
                from scraper_lequipe import scrape_team
                data = scrape_team(name, force_refresh=force_refresh)
                st.session_state[f"{key_prefix}_w"]  = int(data["wins"])
                st.session_state[f"{key_prefix}_d"]  = int(data["draws"])
                st.session_state[f"{key_prefix}_l"]  = int(data["losses"])
                st.session_state[f"{key_prefix}_gs"] = float(data["goals_scored_avg"])
                st.session_state[f"{key_prefix}_gc"] = float(data["goals_conceded_avg"])

                # Passes décisives du meilleur passeur (footmercato.net)
                assists_msg = ""
                try:
                    from scraper_footmercato import get_top_assists
                    pd_val, pd_name = get_top_assists(name)
                    if pd_val:
                        st.session_state[f"{key_prefix}_ma"] = pd_val
                        st.session_state[f"{key_prefix}_pa_name"] = pd_name
                        assists_msg = f"  Meilleur passeur : **{pd_name}** ({pd_val} PD)."
                except Exception:
                    pass

                cache_note = ""
                if data.get("from_cache"):
                    age = data.get("cache_age_min", 0)
                    cache_note = f"  *(cache — il y a {age} min)*"
                else:
                    cache_note = "  *(données fraîches)*"

                matches_preview = "\n".join(
                    f"• {m['date']}  {m['home_team']} {m['score_home']}-{m['score_away']} {m['away_team']}"
                    for m in data.get("matches", [])
                )
                st.success(
                    f"✅ **{data['team_name']}** — {data['nb_matches']} matchs (lequipe.fr).{cache_note}"
                    + rank_msg + assists_msg + "\n\n"
                    + matches_preview + "\n\n"
                    "Vérifiez la présence du buteur/passeur si besoin."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lequipe.fr : {e}")


def render_team_search(key_prefix: str, is_international: bool):
    if not API_AVAILABLE:
        return
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "🔍 Rechercher une équipe",
            key=f"{key_prefix}_search_query",
            placeholder="Ex: Paris Saint-Germain, France..."
        )
    with col_btn:
        st.write("")
        search_clicked = st.button("Rechercher", key=f"{key_prefix}_search_btn")

    if search_clicked and query.strip():
        with st.spinner("Recherche en cours..."):
            try:
                results = search_team(query.strip())
                if results:
                    if is_international:
                        results = [r for r in results if r.get("national")] or results
                    else:
                        results = [r for r in results if not r.get("national")] or results
                    st.session_state[f"{key_prefix}_search_results"] = results[:5]
                else:
                    st.warning("Aucune équipe trouvée.")
            except Exception as e:
                st.error(str(e))

    results = st.session_state.get(f"{key_prefix}_search_results", [])
    if results:
        options = {f"{r['name']} ({r['country']})": r for r in results}
        choice = st.selectbox(
            "Sélectionner l'équipe",
            list(options.keys()),
            key=f"{key_prefix}_search_choice"
        )
        if st.button("✅ Charger les stats", key=f"{key_prefix}_load_btn"):
            team_info = options[choice]
            with st.spinner(f"Récupération des stats pour {team_info['name']}..."):
                try:
                    data = fetch_team_data(
                        team_info["id"],
                        team_info["name"],
                        is_national=team_info["national"]
                    )
                    mapping = {
                        "wins": f"{key_prefix}_w",
                        "draws": f"{key_prefix}_d",
                        "losses": f"{key_prefix}_l",
                        "goals_scored": f"{key_prefix}_gs",
                        "goals_conceded": f"{key_prefix}_gc",
                        "max_goals_by_player": f"{key_prefix}_mg",
                        "max_assists_by_player": f"{key_prefix}_ma",
                    }
                    for data_key, st_key in mapping.items():
                        if data_key in data and data[data_key] is not None:
                            st.session_state[st_key] = data[data_key]
                    if "top_scorer_present" in data:
                        st.session_state[f"{key_prefix}_sp"] = bool(data["top_scorer_present"])
                    if "top_assist_present" in data:
                        st.session_state[f"{key_prefix}_ap"] = bool(data["top_assist_present"])

                    scorer = data.get("top_scorer_name", "")
                    passer = data.get("top_assist_name", "")
                    msg = f"✅ Stats chargées pour **{team_info['name']}**."
                    if scorer:
                        msg += f" Buteur clé : {scorer}."
                    if passer:
                        msg += f" Passeur clé : {passer}."
                    msg += " Vérifiez ensuite leur présence et entrez le rang manuellement."
                    st.success(msg)
                    st.session_state.pop(f"{key_prefix}_search_results", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors du chargement : {e}")


def render_team_form(side_label: str, key_prefix: str, is_international: bool):
    st.subheader(side_label)
    render_lequipe_search(key_prefix, is_international=is_international)
    render_team_search(key_prefix, is_international)
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wins = st.number_input("Victoires", 0, 10, key=f"{key_prefix}_w")
    with col_b:
        draws = st.number_input("Nuls", 0, 10, key=f"{key_prefix}_d")
    with col_c:
        losses = st.number_input("Défaites", 0, 10, key=f"{key_prefix}_l")

    nb = wins + draws + losses
    st.caption(f"Forme sur les {nb} derniers matchs toutes compétitions" if nb > 0 else "Forme sur les derniers matchs toutes compétitions")

    col_d, col_e = st.columns(2)
    with col_d:
        goals_scored = st.number_input("Moy. buts marqués / match", 0.0, 15.0, step=0.1, key=f"{key_prefix}_gs")
    with col_e:
        goals_conceded = st.number_input("Moy. buts encaissés / match", 0.0, 15.0, step=0.1, key=f"{key_prefix}_gc")

    col_f, col_g = st.columns(2)
    with col_f:
        max_goals = st.number_input("Max buts d'un joueur (5 matchs)", 0, 15, key=f"{key_prefix}_mg")
    with col_g:
        max_assists = st.number_input("Passes décisives du meilleur passeur (saison)", 0, 30, key=f"{key_prefix}_ma")
        pa_name = st.session_state.get(f"{key_prefix}_pa_name", "")
        if pa_name:
            st.caption(f"Meilleur passeur : **{pa_name}**")

    col_h, col_i = st.columns(2)
    with col_h:
        scorer_present = st.checkbox(
            "Meilleur buteur disponible ?",
            value=st.session_state.get(f"{key_prefix}_sp", True),
            key=f"{key_prefix}_sp"
        )
    with col_i:
        assist_present = st.checkbox(
            "Meilleur passeur disponible ?",
            value=st.session_state.get(f"{key_prefix}_ap", True),
            key=f"{key_prefix}_ap"
        )

    if is_international:
        rank = st.number_input(
            "Classement FIFA (0 = non classé)", 0, 500,
            value=st.session_state.get(f"{key_prefix}_rank", 50),
            key=f"{key_prefix}_rank"
        )
        team = {
            "wins": wins, "draws": draws, "losses": losses,
            "goals_scored": goals_scored, "goals_conceded": goals_conceded,
            "max_goals_by_player": max_goals,
            "max_assists_by_player": max_assists,
            "top_scorer_present": scorer_present,
            "top_assist_present": assist_present,
            "fifa_rank": rank,
        }
    else:
        rank = st.number_input(
            "Classement UEFA (0 = non classé)", 0, 500,
            value=st.session_state.get(f"{key_prefix}_rank", 50),
            key=f"{key_prefix}_rank"
        )
        team = {
            "wins": wins, "draws": draws, "losses": losses,
            "goals_scored": goals_scored, "goals_conceded": goals_conceded,
            "max_goals_by_player": max_goals,
            "max_assists_by_player": max_assists,
            "top_scorer_present": scorer_present,
            "top_assist_present": assist_present,
            "uefa_rank": rank,
        }
    return team

def render_common_analysis(home, away, key_prefix, competition_label):
    st.markdown("---")
    st.markdown("### 💰 Cotes bookmaker")

    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Cote 1", min_value=1.01, value=2.00, key=f"{key_prefix}_odd_1")
    with c2:
        odd_draw = st.number_input("Cote N", min_value=1.01, value=3.20, key=f"{key_prefix}_odd_n")
    with c3:
        odd_away = st.number_input("Cote 2", min_value=1.01, value=2.00, key=f"{key_prefix}_odd_2")

    c4, c5, c6, c7, c8 = st.columns(5)
    with c4:
        odd_1n = st.number_input("Cote 1N", min_value=1.01, value=1.40, key=f"{key_prefix}_odd_1n")
    with c5:
        odd_12 = st.number_input("Cote 12", min_value=1.01, value=1.40, key=f"{key_prefix}_odd_12")
    with c6:
        odd_2n = st.number_input("Cote 2N", min_value=1.01, value=1.60, key=f"{key_prefix}_odd_2n")
    with c7:
        odd_over25 = st.number_input("Cote Over 2.5", min_value=1.01, value=1.90, key=f"{key_prefix}_odd_o25")
    with c8:
        odd_btts = st.number_input("Cote BTTS", min_value=1.01, value=1.85, key=f"{key_prefix}_odd_btts")

    if st.button("🔍 Analyser", key=f"{key_prefix}_analyse"):
        try:
            res = predict_match(home, away)
            index = confidence_index(res, home, away)

            home_power = get_rank_power(home)
            away_power = get_rank_power(away)

            adj_home = adjust_expected_goals(
                home["goals_scored"],
                home["max_goals_by_player"],
                home["max_assists_by_player"],
                home["top_scorer_present"],
                home["top_assist_present"],
                home_power
            )
            adj_away = adjust_expected_goals(
                away["goals_scored"],
                away["max_goals_by_player"],
                away["max_assists_by_player"],
                away["top_scorer_present"],
                away["top_assist_present"],
                away_power
            )

            best_score, _ = predict_scores(adj_home, adj_away)
            goal_markets = calculate_goal_markets(adj_home, adj_away)

            v_home = value_bet(res["home_prob"], odd_home)
            v_draw = value_bet(res["draw_prob"], odd_draw)
            v_away = value_bet(res["away_prob"], odd_away)
            v_over25 = value_bet(goal_markets["over25"], odd_over25)
            v_btts = value_bet(goal_markets["btts"], odd_btts)

            prob_1n = round(res["home_prob"] + res["draw_prob"], 1)
            prob_12 = round(res["home_prob"] + res["away_prob"], 1)
            prob_2n = round(res["draw_prob"] + res["away_prob"], 1)
            v_1n = value_bet(prob_1n, odd_1n)
            v_12 = value_bet(prob_12, odd_12)
            v_2n = value_bet(prob_2n, odd_2n)

            best_financial = max([
                ("1",        res["home_prob"],       v_home),
                ("N",        res["draw_prob"],        v_draw),
                ("2",        res["away_prob"],        v_away),
                ("1N",       prob_1n,                v_1n),
                ("12",       prob_12,                v_12),
                ("2N",       prob_2n,                v_2n),
                ("Over 2.5", goal_markets["over25"], v_over25),
                ("BTTS",     goal_markets["btts"],   v_btts),
            ], key=lambda x: x[2][2])

            reco = final_reco(index, best_financial[2][2])
            score_str = f"{best_score['home_goals']}-{best_score['away_goals']} ({round(best_score['probability']*100, 2)}%)"

            st.markdown("---")
            st.subheader("📊 Probabilités")
            col1, col2, col3 = st.columns(3)
            col1.metric("1 — Domicile", f"{res['home_prob']}%")
            col2.metric("N — Nul", f"{res['draw_prob']}%")
            col3.metric("2 — Extérieur", f"{res['away_prob']}%")

            st.markdown("---")
            st.subheader("⚽ Marchés de buts")
            col4, col5, col6 = st.columns(3)
            col4.metric("Score probable", score_str)
            col5.metric("Over 2.5", f"{goal_markets['over25']}%")
            col6.metric("BTTS", f"{goal_markets['btts']}%")

            st.markdown("---")
            st.subheader("💰 Value Bet")
            st.table({
                "Marché":      ["1",         "N",         "2",         "1N",    "12",    "2N",    "Over 2.5",            "BTTS"],
                "Modèle (%)":  [res["home_prob"], res["draw_prob"], res["away_prob"], prob_1n, prob_12, prob_2n, goal_markets["over25"], goal_markets["btts"]],
                "Book (%)":    [v_home[1],   v_draw[1],   v_away[1],   v_1n[1], v_12[1], v_2n[1], v_over25[1],           v_btts[1]],
                "Edge":        [v_home[2],   v_draw[2],   v_away[2],   v_1n[2], v_12[2], v_2n[2], v_over25[2],           v_btts[2]],
                "Signal":      [v_home[0],   v_draw[0],   v_away[0],   v_1n[0], v_12[0], v_2n[0], v_over25[0],           v_btts[0]],
            })

            st.markdown("---")
            st.subheader("🎯 Synthèse")
            st.write(f"**Résultat le plus probable :** {res['prediction']} — {res['label']}")
            st.write(f"**Pari le plus intéressant :** {format_market_label(best_financial[0])}")
            st.write(f"**Verdict global :** {reco}")
            st.write(f"**Indice de confiance :** {index} / 100")

            # --- Guide parieur ---
            st.markdown("---")
            st.subheader("📋 Guide parieur")

            all_markets_guide = [
                ("1",    "Victoire domicile",        res["home_prob"], v_home[2]),
                ("N",    "Match nul",                res["draw_prob"],  v_draw[2]),
                ("2",    "Victoire extérieur",       res["away_prob"],  v_away[2]),
                ("1N",   "Domicile ou nul",          prob_1n,           v_1n[2]),
                ("12",   "Domicile ou extérieur",    prob_12,           v_12[2]),
                ("2N",   "Extérieur ou nul",         prob_2n,           v_2n[2]),
                ("Over 2.5", "Over 2.5 buts",        goal_markets["over25"], v_over25[2]),
                ("BTTS", "Les 2 équipes marquent",   goal_markets["btts"],   v_btts[2]),
            ]

            # Pari le plus sûr : probabilité modèle la plus élevée
            safest = max(all_markets_guide, key=lambda x: x[2])

            # Pari le plus rentable : edge le plus élevé
            most_valuable = max(all_markets_guide, key=lambda x: x[3])

            # Combinaisons : marchés à edge positif compatibles
            # On exclut les doubles chances de la combinaison avec les marchés de résultat simples
            # (1N/12/2N sont déjà des combinaisons résultat, pas compatible avec 1/N/2)
            result_markets_pos = [m for m in all_markets_guide if m[0] in ("1", "N", "2") and m[3] > 0]
            dc_markets_pos     = [m for m in all_markets_guide if m[0] in ("1N", "12", "2N") and m[3] > 0]
            goal_markets_pos   = [m for m in all_markets_guide if m[0] in ("Over 2.5", "BTTS") and m[3] > 0]

            combos = []
            for rm in result_markets_pos:
                for gm in goal_markets_pos:
                    combos.append(f"{rm[1]} + {gm[1]}")
            for dc in dc_markets_pos:
                for gm in goal_markets_pos:
                    combos.append(f"{dc[1]} + {gm[1]}")
            if len(goal_markets_pos) == 2:
                combos.append("Over 2.5 + BTTS")
                for rm in result_markets_pos:
                    combos.append(f"{rm[1]} + Over 2.5 + BTTS")
                for dc in dc_markets_pos:
                    combos.append(f"{dc[1]} + Over 2.5 + BTTS")

            col_s, col_v = st.columns(2)

            with col_s:
                st.markdown("**Le pari le plus sûr**")
                st.info(
                    f"**{safest[1]}**\n\n"
                    f"Probabilité modèle : **{safest[2]}%**\n\n"
                    f"C'est le résultat que le modèle juge le plus probable sur ce match. "
                    f"Privilégiez ce marché si vous cherchez à limiter le risque."
                )

            with col_v:
                st.markdown("**Le pari le plus rentable**")
                edge_val = most_valuable[3]
                if edge_val > 0:
                    st.success(
                        f"**{most_valuable[1]}**\n\n"
                        f"Edge : **+{edge_val}%** par rapport à la cote du bookmaker\n\n"
                        f"C'est le marché où vous avez le plus grand avantage statistique. "
                        f"Un edge positif signifie que le bookmaker sous-évalue ce résultat."
                    )
                else:
                    st.warning(
                        f"**{most_valuable[1]}**\n\n"
                        f"Edge : **{edge_val}%**\n\n"
                        f"Aucun marché ne présente d'avantage net sur ce match. "
                        f"Les cotes reflètent bien les probabilités — pariez avec prudence."
                    )

            st.markdown("**Combinaisons suggérées**")
            if combos:
                combos_uniq = list(dict.fromkeys(combos))  # dédoublonnage ordre préservé
                lines = "\n".join(f"- {c}" for c in combos_uniq[:6])
                st.success(
                    f"Les marchés suivants présentent tous un edge positif et peuvent être combinés :\n\n"
                    + lines + "\n\n"
                    "⚠️ En combiné, les probabilités se multiplient — assurez-vous que chaque sélection est solide individuellement avant de les assembler."
                )
            else:
                st.warning(
                    "Aucune combinaison recommandée sur ce match : moins de deux marchés présentent un edge positif simultanément. "
                    "Combiner des marchés sans edge vous désavantage statistiquement."
                )

            # Combinaison la plus en phase avec la réalité
            st.markdown("**La combinaison la plus en phase avec la réalité**")

            # Résultat le plus probable (parmi 1/N/2)
            result_options = [
                ("1", "Victoire domicile", res["home_prob"]),
                ("N", "Match nul",         res["draw_prob"]),
                ("2", "Victoire extérieur",res["away_prob"]),
            ]
            most_likely_result = max(result_options, key=lambda x: x[2])

            # Marchés de buts avec probabilité > 50 %
            likely_goal_markets = []
            if goal_markets["over25"] >= 50:
                likely_goal_markets.append(("Over 2.5", goal_markets["over25"]))
            if goal_markets["btts"] >= 50:
                likely_goal_markets.append(("BTTS", goal_markets["btts"]))

            # Construction de la combinaison réaliste
            realistic_parts = [most_likely_result[1]]
            for label, _ in likely_goal_markets:
                realistic_parts.append(label)

            realistic_combo = " + ".join(realistic_parts)

            # Probabilité jointe estimée (produit des probabilités indépendantes)
            joint_prob = most_likely_result[2] / 100
            for _, prob in likely_goal_markets:
                joint_prob *= prob / 100
            joint_prob_pct = round(joint_prob * 100, 1)

            if len(realistic_parts) == 1:
                combo_note = (
                    f"Seul le résultat est clairement en faveur d'un scénario : "
                    f"**{most_likely_result[1]}** ({most_likely_result[2]}%). "
                    f"Les marchés de buts sont trop incertains (< 50%) pour enrichir la combinaison."
                )
                st.info(f"**{realistic_combo}**\n\n{combo_note}")
            else:
                combo_note = (
                    f"Le modèle prédit **{most_likely_result[1]}** ({most_likely_result[2]}%) "
                    + " et ".join(f"**{label}** ({prob}%)" for label, prob in likely_goal_markets)
                    + f". Probabilité jointe estimée : **{joint_prob_pct}%**.\n\n"
                    "Cette combinaison reflète le scénario jugé le plus cohérent avec l'ensemble des données — "
                    "indépendamment de la valeur financière."
                )
                st.info(f"**{realistic_combo}**\n\n{combo_note}")

            rang_h = home.get("uefa_rank", home.get("fifa_rank", 0))
            rang_a = away.get("uefa_rank", away.get("fifa_rank", 0))

            # Récupère les noms d'équipes saisis dans les champs lequipe.fr
            home_name = st.session_state.get(f"{key_prefix}_h_lequipe_name", "Domicile") or "Domicile"
            away_name = st.session_state.get(f"{key_prefix}_a_lequipe_name", "Extérieur") or "Extérieur"

            log_analyse({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "competition": competition_label,
                "equipe_domicile": home_name,
                "equipe_exterieur": away_name,
                "wins_h": home["wins"], "draws_h": home["draws"], "losses_h": home["losses"],
                "goals_scored_h": home["goals_scored"], "goals_conceded_h": home["goals_conceded"],
                "max_goals_player_h": home["max_goals_by_player"],
                "max_assists_player_h": home["max_assists_by_player"],
                "top_scorer_present_h": home["top_scorer_present"],
                "top_assist_present_h": home["top_assist_present"], "rang_h": rang_h,
                "wins_a": away["wins"], "draws_a": away["draws"], "losses_a": away["losses"],
                "goals_scored_a": away["goals_scored"], "goals_conceded_a": away["goals_conceded"],
                "max_goals_player_a": away["max_goals_by_player"],
                "max_assists_player_a": away["max_assists_by_player"],
                "top_scorer_present_a": away["top_scorer_present"],
                "top_assist_present_a": away["top_assist_present"], "rang_a": rang_a,
                "cote_1": odd_home, "cote_n": odd_draw, "cote_2": odd_away,
                "cote_over25": odd_over25, "cote_btts": odd_btts,
                "prob_domicile": res["home_prob"], "prob_nul": res["draw_prob"],
                "prob_exterieur": res["away_prob"],
                "prediction": res["prediction"], "indice_confiance": index,
                "score_probable": score_str,
                "over25_pct": goal_markets["over25"], "btts_pct": goal_markets["btts"],
                "value_1": v_home[0], "edge_1": v_home[2],
                "value_n": v_draw[0], "edge_n": v_draw[2],
                "value_2": v_away[0], "edge_2": v_away[2],
                "value_over25": v_over25[0], "edge_over25": v_over25[2],
                "value_btts": v_btts[0], "edge_btts": v_btts[2],
                "meilleur_marche": best_financial[0], "verdict": reco,
                "pari_realise": "", "resultat_reel": "", "score_reel": "",
            })
            st.success("✅ Analyse sauvegardée dans historique.csv")
            if os.path.exists(HISTORIQUE_PATH):
                with open(HISTORIQUE_PATH, "rb") as f_csv:
                    st.download_button(
                        label="📥 Télécharger l'historique CSV",
                        data=f_csv,
                        file_name="historique_paris.csv",
                        mime="text/csv",
                        key=f"dl_{key_prefix}_{datetime.now().strftime('%H%M%S')}",
                    )

        except ValueError as e:
            st.error(str(e))
        except KeyError as e:
            st.error(f"Clé manquante dans les données : {e}")

def _score_to_result(score_str: str) -> str | None:
    """'2-1' → '1', '1-1' → 'N', '0-2' → '2'. Retourne None si invalide."""
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", str(score_str).strip())
    if not m:
        return None
    h, a = int(m.group(1)), int(m.group(2))
    if h > a:
        return "1"
    if h == a:
        return "N"
    return "2"


def _score_btts(score_str: str) -> bool | None:
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", str(score_str).strip())
    if not m:
        return None
    return int(m.group(1)) > 0 and int(m.group(2)) > 0


def _score_over25(score_str: str) -> bool | None:
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", str(score_str).strip())
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) > 2


def _market_won(market: str, score_str: str) -> bool | None:
    """Retourne True si le marché 'market' était gagnant pour le score donné."""
    if not score_str or str(score_str).strip() == "":
        return None
    result = _score_to_result(score_str)
    if market in ("1", "N", "2"):
        return result == market if result else None
    if market == "1N":
        return result in ("1", "N") if result else None
    if market == "12":
        return result in ("1", "2") if result else None
    if market == "2N":
        return result in ("2", "N") if result else None
    if market == "Over 2.5":
        return _score_over25(score_str)
    if market == "BTTS":
        return _score_btts(score_str)
    return None


def _any_market_won(marche_str: str, score_str: str) -> bool | None:
    """
    Utilisé pour 'meilleur_marche' : plusieurs suggestions séparées par '+'.
    Retourne True si AU MOINS UN des marchés suggérés est gagnant.
    """
    marche_str = str(marche_str).strip()
    if not marche_str or not str(score_str).strip():
        return None
    markets = [m.strip() for m in re.split(r"[+,]", marche_str) if m.strip()]
    if not markets:
        return None
    results = [_market_won(m, score_str) for m in markets]
    known = [r for r in results if r is not None]
    if not known:
        return None
    return any(known)


def _combo_won(pari_str: str, score_str: str) -> bool | None:
    """
    Gère les paris simples et combinés (ex: '1 + Over 2.5').
    Un combiné est gagnant seulement si TOUS les marchés sont gagnants.
    Retourne None si le champ est vide ou le score inconnu.
    """
    pari_str = str(pari_str).strip()
    if not pari_str or not str(score_str).strip():
        return None
    # Découpe sur '+' ou ',' en nettoyant chaque token
    markets = [m.strip() for m in re.split(r"[+,]", pari_str) if m.strip()]
    if not markets:
        return None
    results = [_market_won(m, score_str) for m in markets]
    # Si l'un des marchés est inconnu → on ne peut pas trancher
    if any(r is None for r in results):
        return None
    return all(results)


def _load_historique() -> pd.DataFrame:
    if not os.path.exists(HISTORIQUE_PATH):
        return pd.DataFrame(columns=HISTORIQUE_COLUMNS)
    df = pd.read_csv(HISTORIQUE_PATH, dtype=str)
    # Ajoute les colonnes manquantes (rétrocompatibilité)
    for col in HISTORIQUE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _save_historique(df: pd.DataFrame):
    df.to_csv(HISTORIQUE_PATH, index=False)


ANALYSE_PATH = "analyse_manuelle.csv"
ANALYSE_COLUMNS = [
    "date", "competition", "equipe_domicile", "equipe_exterieur",
    "prediction", "meilleur_marche", "pari_realise", "score_reel",
]

MARKET_OPTIONS = ["", "1", "N", "2", "1N", "12", "2N", "Over 2.5", "BTTS"]


def _load_analyse() -> pd.DataFrame:
    if not os.path.exists(ANALYSE_PATH):
        return pd.DataFrame(columns=ANALYSE_COLUMNS)
    df = pd.read_csv(ANALYSE_PATH, dtype=str).fillna("")
    for col in ANALYSE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ANALYSE_COLUMNS]


def _save_analyse(df: pd.DataFrame):
    df[ANALYSE_COLUMNS].to_csv(ANALYSE_PATH, index=False)


def render_analyse_tab():
    st.header("📊 Analyse des prédictions")

    # ----------------------------------------------------------------
    # Import CSV (persistance entre sessions / téléphone)
    # ----------------------------------------------------------------
    with st.expander("📂 Importer un fichier existant", expanded=not os.path.exists(ANALYSE_PATH)):
        st.caption(
            "Sur téléphone ou Streamlit Cloud, les données ne sont pas conservées entre les sessions. "
            "**Importez votre fichier** au début de chaque session, puis **téléchargez-le** à la fin pour le sauvegarder."
        )
        uploaded = st.file_uploader(
            "Charger analyse_predictions.csv",
            type="csv",
            key="analyse_upload",
        )
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded, dtype=str).fillna("")
                for col in ANALYSE_COLUMNS:
                    if col not in df_up.columns:
                        df_up[col] = ""
                _save_analyse(df_up)
                st.success("✅ Fichier importé et sauvegardé.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur à l'import : {e}")

    df = _load_analyse()

    # ----------------------------------------------------------------
    # Section 1 : tableau de saisie manuelle
    # ----------------------------------------------------------------
    st.subheader("📝 Tableau des matchs")

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.TextColumn(
                "Date", help="Ex : 2026-04-20", width="small"
            ),
            "competition": st.column_config.TextColumn(
                "Compétition", help="Ex : Ligue 1, Liga…", width="small"
            ),
            "equipe_domicile": st.column_config.TextColumn(
                "Domicile", width="medium"
            ),
            "equipe_exterieur": st.column_config.TextColumn(
                "Extérieur", width="medium"
            ),
            "prediction": st.column_config.SelectboxColumn(
                "Résultat le plus probable",
                options=MARKET_OPTIONS,
                help="Résultat le plus probable donné par l'outil",
                width="small",
            ),
            "meilleur_marche": st.column_config.TextColumn(
                "Meilleur(s) marché(s)",
                help="Ex : 1  /  Over 2.5  /  1 + Over 2.5  (n°1 + n°2 de l'outil)",
                width="medium",
            ),
            "pari_realise": st.column_config.TextColumn(
                "Pari(s) réalisé(s)",
                help="Ex : 1  /  Over 2.5  /  1 + BTTS  /  2 + Over 2.5",
                width="medium",
            ),
            "score_reel": st.column_config.TextColumn(
                "Score réel",
                help="Format : 2-1 (domicile-extérieur)",
                width="small",
            ),
        },
        key="analyse_editor",
    )

    col_save, col_dl = st.columns([2, 3])
    with col_save:
        if st.button("💾 Sauvegarder", type="primary"):
            _save_analyse(edited)
            st.success("✅ Tableau sauvegardé.")
            st.rerun()
    with col_dl:
        if os.path.exists(ANALYSE_PATH):
            with open(ANALYSE_PATH, "rb") as f_csv:
                st.download_button(
                    label="📥 Télécharger pour sauvegarder",
                    data=f_csv,
                    file_name="analyse_predictions.csv",
                    mime="text/csv",
                    help="Téléchargez ce fichier pour le réimporter lors de votre prochaine session",
                )

    # ----------------------------------------------------------------
    # Section 2 : statistiques
    # ----------------------------------------------------------------
    df_known = edited[edited["score_reel"].apply(lambda s: bool(str(s).strip()))].copy()

    if df_known.empty:
        st.info("Renseignez au moins un score réel pour voir les statistiques.")
        return

    df_known["resultat_reel"] = df_known["score_reel"].apply(lambda s: _score_to_result(s) or "")
    df_known["pred_ok"]       = df_known.apply(
        lambda r: bool(_market_won(r["prediction"], r["score_reel"])) if r["prediction"] else False, axis=1
    )
    df_known["meilleur_ok"]   = df_known.apply(
        lambda r: _any_market_won(r["meilleur_marche"], r["score_reel"]), axis=1
    )
    df_known["pari_ok"]       = df_known.apply(
        lambda r: _combo_won(r["pari_realise"], r["score_reel"]),
        axis=1,
    )

    n_total = len(df_known)

    st.markdown("---")
    st.subheader("📈 Statistiques globales")

    pred_ok_n       = int(df_known["pred_ok"].sum())
    meilleur_series = df_known["meilleur_ok"].dropna()
    meilleur_ok_n   = int(meilleur_series.sum())
    meilleur_denom  = len(meilleur_series)
    pari_series     = df_known["pari_ok"].dropna()
    pari_ok_n       = int(pari_series.sum())
    pari_denom      = len(pari_series)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Matchs renseignés", n_total)
    col_s2.metric(
        "Résultat le plus probable correct",
        f"{round(pred_ok_n / n_total * 100, 1)}%" if n_total else "—",
        f"{pred_ok_n}/{n_total}",
    )
    col_s3.metric(
        "Meilleur marché gagnant",
        f"{round(meilleur_ok_n / meilleur_denom * 100, 1)}%" if meilleur_denom else "—",
        f"{meilleur_ok_n}/{meilleur_denom}" if meilleur_denom else "",
    )
    col_s4.metric(
        "Pari réalisé gagnant",
        f"{round(pari_ok_n / pari_denom * 100, 1)}%" if pari_denom else "—",
        f"{pari_ok_n}/{pari_denom}" if pari_denom else "",
    )

    # ----------------------------------------------------------------
    # Section 3 : performance par marché prédit
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Performance par marché prédit (meilleur marché de l'outil)")

    # Explose les entrées multi-marchés pour compter chaque marché individuellement
    market_rows = []
    for _, row in df_known.iterrows():
        markets = [m.strip() for m in re.split(r"[+,]", str(row["meilleur_marche"])) if m.strip()]
        for m in markets:
            market_rows.append({"market": m, "score_reel": row["score_reel"]})

    market_stats = []
    for market in ["1", "N", "2", "1N", "12", "2N", "Over 2.5", "BTTS"]:
        subset = [r for r in market_rows if r["market"] == market]
        if not subset:
            continue
        wins = [_market_won(market, r["score_reel"]) for r in subset]
        wins_known = [w for w in wins if w is not None]
        if not wins_known:
            continue
        market_stats.append({
            "Marché": format_market_label(market),
            "Prédictions": len(subset),
            "Gagnant": int(sum(wins_known)),
            "Taux réussite": f"{round(sum(wins_known) / len(wins_known) * 100, 1)}%",
        })

    if market_stats:
        st.dataframe(pd.DataFrame(market_stats), use_container_width=True, hide_index=True)
    else:
        st.info("Renseignez la colonne 'Meilleur marché' pour voir cette analyse.")

    # ----------------------------------------------------------------
    # Section 4 : analyse statistique et fiabilité
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Analyse de fiabilité")

    if n_total < 3:
        st.info("Ajoutez au minimum 3 matchs avec score réel pour générer l'analyse.")
    else:
        # --- Fiabilité du résultat le plus probable ---
        st.markdown("**Résultat le plus probable — fiabilité globale**")
        taux_pred = pred_ok_n / n_total * 100

        # Intervalle de confiance approximatif (Wilson simplifié)
        import math
        z = 1.96
        p = pred_ok_n / n_total
        margin = z * math.sqrt(p * (1 - p) / n_total) * 100
        ic_low  = max(0, taux_pred - margin)
        ic_high = min(100, taux_pred + margin)

        if taux_pred >= 60:
            fiab_label = "Bonne fiabilité"
            fiab_color = "normal"
        elif taux_pred >= 45:
            fiab_label = "Fiabilité moyenne"
            fiab_color = "off"
        else:
            fiab_label = "Fiabilité faible"
            fiab_color = "inverse"

        st.metric(
            fiab_label,
            f"{round(taux_pred, 1)}%",
            f"IC 95% : [{round(ic_low,1)}% – {round(ic_high,1)}%]  ·  {pred_ok_n}/{n_total} corrects",
            delta_color=fiab_color,
        )

        # Détail par type de résultat prédit
        res_types = [
            ("1",  "Victoire domicile",    lambda r: r == "1"),
            ("N",  "Nul",                  lambda r: r == "N"),
            ("2",  "Victoire extérieur",   lambda r: r == "2"),
            ("1N", "Domicile ou nul",      lambda r: r in ("1", "N")),
            ("12", "Domicile ou extérieur",lambda r: r in ("1", "2")),
            ("2N", "Extérieur ou nul",     lambda r: r in ("2", "N")),
        ]
        rows_res = []
        for code, label, check_fn in res_types:
            subset_r = df_known[df_known["prediction"] == code]
            if len(subset_r) == 0:
                continue
            ok = int(subset_r["resultat_reel"].apply(lambda r: bool(check_fn(r)) if r else False).sum())
            total = len(subset_r)
            taux_r = round(ok / total * 100, 1)
            rows_res.append({
                "Résultat prédit": label,
                "Prédictions": total,
                "Corrects": ok,
                "Taux": f"{taux_r}%",
                "Fiabilité": "✅ Fiable" if taux_r >= 60 else ("⚠️ Moyen" if taux_r >= 40 else "❌ Faible"),
            })
        if rows_res:
            st.dataframe(pd.DataFrame(rows_res), use_container_width=True, hide_index=True)

        # --- Fiabilité par marché (meilleur marché suggéré) ---
        if market_stats:
            st.markdown("**Marchés suggérés — classement par fiabilité**")
            df_mkt = pd.DataFrame(market_stats).copy()
            df_mkt["_taux_num"] = df_mkt["Taux réussite"].str.replace("%", "").astype(float)
            df_mkt["Fiabilité"] = df_mkt["_taux_num"].apply(
                lambda t: "✅ Fiable" if t >= 60 else ("⚠️ Moyen" if t >= 40 else "❌ Faible")
            )
            df_mkt["Échantillon"] = df_mkt["Prédictions"].apply(
                lambda n: "⚠️ Faible (<5)" if n < 5 else ("Moyen (5-10)" if n < 10 else "Solide (≥10)")
            )
            df_mkt = df_mkt.sort_values("_taux_num", ascending=False).drop(columns=["_taux_num"])
            st.dataframe(df_mkt, use_container_width=True, hide_index=True)

            best = max(market_stats, key=lambda x: float(x["Taux réussite"].replace("%", "")))
            worst = min(market_stats, key=lambda x: float(x["Taux réussite"].replace("%", "")))

            col_b1, col_b2 = st.columns(2)
            col_b1.metric(
                "Marché le plus fiable",
                best["Marché"],
                f"{best['Taux réussite']} sur {best['Prédictions']} matchs",
            )
            if len(market_stats) > 1:
                col_b2.metric(
                    "Marché le moins fiable",
                    worst["Marché"],
                    f"{worst['Taux réussite']} sur {worst['Prédictions']} matchs",
                    delta_color="inverse",
                )

        # --- Comparaison : meilleur marché outil vs pari réalisé ---
        if meilleur_denom >= 3 and pari_denom >= 3:
            st.markdown("**Suggestions outil vs Paris réalisés**")
            taux_meilleur = round(meilleur_ok_n / meilleur_denom * 100, 1)
            taux_pari     = round(pari_ok_n / pari_denom * 100, 1)
            diff = taux_pari - taux_meilleur

            col_c1, col_c2 = st.columns(2)
            col_c1.metric(
                "Meilleur marché outil",
                f"{taux_meilleur}%",
                f"{meilleur_ok_n}/{meilleur_denom} gagnants",
            )
            col_c2.metric(
                "Paris réalisés",
                f"{taux_pari}%",
                f"{pari_ok_n}/{pari_denom} gagnants",
                delta_color="normal" if diff >= 0 else "inverse",
            )

            if abs(diff) >= 5:
                if diff > 0:
                    st.markdown(
                        f"> Vos paris réalisés ({taux_pari}%) surpassent les suggestions de l'outil ({taux_meilleur}%) "
                        f"de **{round(diff, 1)} points**. Votre sélection personnelle apporte de la valeur."
                    )
                else:
                    st.markdown(
                        f"> Les suggestions de l'outil ({taux_meilleur}%) surpassent vos paris réalisés ({taux_pari}%) "
                        f"de **{round(-diff, 1)} points**. Suivre l'outil plus fidèlement serait plus rentable."
                    )



# ============================================================
# MAIN
# ============================================================
st.title("⚽ Mode Parieur")

tab1, tab2, tab3 = st.tabs(["🏟️ Clubs", "🌍 Sélections internationales", "📊 Analyse"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        home_club = render_team_form("Domicile", "club_h", is_international=False)
    with col2:
        away_club = render_team_form("Extérieur", "club_a", is_international=False)
    render_common_analysis(home_club, away_club, "club", "Clubs")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        home_int = render_team_form("Domicile", "int_h", is_international=True)
    with col2:
        away_int = render_team_form("Extérieur", "int_a", is_international=True)
    render_common_analysis(home_int, away_int, "int", "Sélections")

with tab3:
    render_analyse_tab()

