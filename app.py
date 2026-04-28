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

st.set_page_config(page_title="BetPredict Pro", page_icon="⚽", layout="wide")


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
    css_class = "team-header-home" if side_label == "Domicile" else "team-header-away"
    icon = "🏠" if side_label == "Domicile" else "✈️"
    st.markdown(f'<div class="{css_class}">{icon} {side_label}</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">💰 Cotes bookmaker</div>', unsafe_allow_html=True)

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
            st.markdown('<div class="section-header">📊 Probabilités</div>', unsafe_allow_html=True)

            def _bar_color(p):
                if p >= 55: return "#00c853"
                if p >= 40: return "#f5a623"
                return "#ff5252"

            col1, col2, col3 = st.columns(3)
            for col, label, prob in [
                (col1, "1 — Domicile",   res["home_prob"]),
                (col2, "N — Nul",        res["draw_prob"]),
                (col3, "2 — Extérieur",  res["away_prob"]),
            ]:
                col.markdown(f"""
                <div class="prob-container">
                  <div class="prob-label">{label}</div>
                  <div class="prob-value">{prob}%</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob}%;background:{_bar_color(prob)};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">⚽ Marchés de buts</div>', unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            col4.markdown(f"""
            <div class="prob-container">
              <div class="prob-label">Score probable</div>
              <div class="prob-value" style="font-size:1.2rem">{score_str}</div>
            </div>""", unsafe_allow_html=True)
            for col, label, prob in [
                (col5, "Over 2.5", goal_markets["over25"]),
                (col6, "BTTS",     goal_markets["btts"]),
            ]:
                col.markdown(f"""
                <div class="prob-container">
                  <div class="prob-label">{label}</div>
                  <div class="prob-value">{prob}%</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob}%;background:{_bar_color(prob)};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">💰 Value Bet</div>', unsafe_allow_html=True)

            vb_rows = [
                ("1",        res["home_prob"], v_home[1],   v_home[2],   v_home[0]),
                ("N",        res["draw_prob"], v_draw[1],   v_draw[2],   v_draw[0]),
                ("2",        res["away_prob"], v_away[1],   v_away[2],   v_away[0]),
                ("1N",       prob_1n,          v_1n[1],     v_1n[2],     v_1n[0]),
                ("12",       prob_12,          v_12[1],     v_12[2],     v_12[0]),
                ("2N",       prob_2n,          v_2n[1],     v_2n[2],     v_2n[0]),
                ("Over 2.5", goal_markets["over25"], v_over25[1], v_over25[2], v_over25[0]),
                ("BTTS",     goal_markets["btts"],   v_btts[1],   v_btts[2],   v_btts[0]),
            ]
            vb_html = '<table class="vb-table"><thead><tr>'
            for h in ["Marché", "Modèle", "Book", "Edge", "Signal"]:
                vb_html += f"<th>{h}</th>"
            vb_html += "</tr></thead><tbody>"
            for market, modele, book, edge, signal in vb_rows:
                edge_cls = "edge-pos" if edge > 0 else ("edge-neg" if edge < 0 else "edge-neu")
                edge_str = f"+{edge}%" if edge > 0 else f"{edge}%"
                sig_cls  = "signal-value" if signal == "Value" else "signal-nv"
                vb_html += (
                    f"<tr>"
                    f"<td><strong>{market}</strong></td>"
                    f"<td>{modele}%</td>"
                    f"<td>{book}%</td>"
                    f'<td class="{edge_cls}">{edge_str}</td>'
                    f'<td class="{sig_cls}">{signal}</td>'
                    f"</tr>"
                )
            vb_html += "</tbody></table>"
            st.markdown(vb_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Synthèse</div>', unsafe_allow_html=True)

            conf_color = "#00c853" if index >= 60 else ("#f5a623" if index >= 40 else "#ff5252")
            st.markdown(f"""
            <div class="synthese-card">
              <div class="synth-row">
                <span class="synth-key">Résultat le plus probable</span>
                <span class="synth-val accent">{res['prediction']} — {res['label']}</span>
              </div>
              <div class="synth-row">
                <span class="synth-key">Meilleur marché</span>
                <span class="synth-val gold">{format_market_label(best_financial[0])}</span>
              </div>
              <div class="synth-row">
                <span class="synth-key">Verdict</span>
                <span class="synth-val">{reco}</span>
              </div>
              <div class="synth-row" style="border-bottom:none;padding-bottom:0">
                <span class="synth-key">Indice de confiance</span>
                <span class="synth-val" style="color:{conf_color}">{index} / 100</span>
              </div>
              <div class="confidence-bar" style="margin-top:10px">
                <div class="confidence-fill" style="width:{index}%;background:{conf_color}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # --- Guide parieur ---
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Guide parieur</div>', unsafe_allow_html=True)

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
                col_s.markdown(f"""
                <div class="guide-card safe">
                  <div class="card-title">🛡️ Le pari le plus sûr</div>
                  <div class="card-main">{safest[1]}</div>
                  <div class="card-sub">Probabilité modèle : <strong>{safest[2]}%</strong><br>
                  Le résultat que le modèle juge le plus probable. Privilégiez ce marché pour limiter le risque.</div>
                </div>""", unsafe_allow_html=True)

            with col_v:
                edge_val = most_valuable[3]
                if edge_val > 0:
                    col_v.markdown(f"""
                    <div class="guide-card value">
                      <div class="card-title">💎 Le pari le plus rentable</div>
                      <div class="card-main">{most_valuable[1]}</div>
                      <div class="card-sub">Edge : <strong>+{edge_val}%</strong> vs le bookmaker<br>
                      Le bookmaker sous-évalue ce résultat — avantage statistique maximal.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    col_v.markdown(f"""
                    <div class="guide-card warning">
                      <div class="card-title">⚠️ Pari le plus rentable</div>
                      <div class="card-main">{most_valuable[1]}</div>
                      <div class="card-sub">Edge : <strong>{edge_val}%</strong><br>
                      Aucun marché ne présente d'avantage net — pariez avec prudence.</div>
                    </div>""", unsafe_allow_html=True)

            if combos:
                combos_uniq = list(dict.fromkeys(combos))
                items = "".join(f"<li>{c}</li>" for c in combos_uniq[:6])
                st.markdown(f"""
                <div class="guide-card value" style="margin-top:10px">
                  <div class="card-title">🔗 Combinaisons suggérées (edge positif)</div>
                  <ul style="color:#b0bec5;margin:6px 0 4px 16px;font-size:0.88rem;line-height:1.8">{items}</ul>
                  <div class="card-sub" style="margin-top:6px">⚠️ En combiné, les probabilités se multiplient — chaque sélection doit être solide individuellement.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="guide-card warning" style="margin-top:10px">
                  <div class="card-title">🔗 Combinaisons</div>
                  <div class="card-sub">Aucune combinaison recommandée : moins de deux marchés présentent un edge positif simultanément.
                  Combiner sans edge vous désavantage statistiquement.</div>
                </div>""", unsafe_allow_html=True)

            # Combinaison la plus en phase avec la réalité
            st.markdown('<div class="card-title" style="color:#90a4ae;font-size:0.78rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-top:14px;margin-bottom:6px">🎯 La combinaison la plus en phase avec la réalité</div>', unsafe_allow_html=True)

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
                    f"Résultat clair : <strong>{most_likely_result[1]}</strong> ({most_likely_result[2]}%). "
                    f"Les marchés de buts sont trop incertains (&lt;50%) pour enrichir la combinaison."
                )
            else:
                parts_desc = f"<strong>{most_likely_result[1]}</strong> ({most_likely_result[2]}%) + "
                parts_desc += " + ".join(f"<strong>{label}</strong> ({prob}%)" for label, prob in likely_goal_markets)
                combo_note = (
                    parts_desc
                    + f" — Proba jointe estimée : <strong>{joint_prob_pct}%</strong>.<br>"
                    "Scénario le plus cohérent avec l'ensemble des données, indépendamment de la valeur financière."
                )
            st.markdown(f"""
            <div class="guide-card safe">
              <div class="card-main">{realistic_combo}</div>
              <div class="card-sub">{combo_note}</div>
            </div>""", unsafe_allow_html=True)

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

# ---- CSS global ----
st.markdown("""
<style>
/* ===== BASE — fond clair ===== */
[data-testid="stAppViewContainer"] {
    background: #f0f4f8;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; }

/* ===== HEADER PRINCIPAL ===== */
.betpredict-header {
    background: linear-gradient(135deg, #0d3349 0%, #1a5c6e 50%, #0d3349 100%);
    border-bottom: 4px solid #00b894;
    padding: 18px 32px 14px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.betpredict-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.betpredict-header .tagline {
    color: #81ecec;
    font-size: 0.82rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 3px;
}
.betpredict-logo { font-size: 2.8rem; line-height: 1; }

/* ===== SECTION HEADERS ===== */
.section-header {
    background: linear-gradient(90deg, #dff0f7 0%, #f0f4f8 100%);
    border-left: 4px solid #0984e3;
    padding: 8px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
    color: #0d3349;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ===== TEAM HEADERS ===== */
.team-header-home {
    background: linear-gradient(135deg, #0d3349 0%, #0984e3 100%);
    border-radius: 10px;
    padding: 11px 16px;
    text-align: center;
    color: #ffffff;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(9,132,227,0.3);
}
.team-header-away {
    background: linear-gradient(135deg, #6c3483 0%, #e84393 100%);
    border-radius: 10px;
    padding: 11px 16px;
    text-align: center;
    color: #ffffff;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(232,67,147,0.3);
}

/* ===== PROBABILITY BARS ===== */
.prob-container {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px;
    margin: 4px 0;
    border: 1px solid #dfe6e9;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.prob-label {
    color: #636e72;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.prob-value {
    color: #2d3436;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
}
.prob-bar-bg {
    background: #dfe6e9;
    border-radius: 4px;
    height: 7px;
    margin-top: 9px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* ===== VALUE BET TABLE ===== */
.vb-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.88rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.vb-table th {
    background: #0d3349;
    color: #b2bec3;
    font-weight: 700;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 14px;
    text-align: left;
}
.vb-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #f0f4f8;
    color: #2d3436;
    font-weight: 500;
}
.vb-table tr:last-child td { border-bottom: none; }
.vb-table tr:hover td { background: #f8fbff; }
.edge-pos { color: #00b894; font-weight: 700; }
.edge-neg { color: #d63031; font-weight: 700; }
.edge-neu { color: #636e72; }
.signal-value { color: #0984e3; font-weight: 700; }
.signal-nv    { color: #d63031; font-weight: 700; }

/* ===== SYNTHESE CARD ===== */
.synthese-card {
    background: #ffffff;
    border: 2px solid #00b894;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    box-shadow: 0 2px 12px rgba(0,184,148,0.12);
}
.synthese-card .synth-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #f0f4f8;
}
.synthese-card .synth-row:last-child { border-bottom: none; }
.synthese-card .synth-key {
    color: #636e72;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.synthese-card .synth-val {
    color: #2d3436;
    font-weight: 700;
    font-size: 0.95rem;
}
.synthese-card .synth-val.accent { color: #0984e3; }
.synthese-card .synth-val.gold   { color: #e17055; }

/* ===== CONFIDENCE BAR ===== */
.confidence-bar {
    background: #dfe6e9;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-top: 10px;
}
.confidence-fill {
    height: 100%;
    border-radius: 6px;
}

/* ===== GUIDE CARDS ===== */
.guide-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin: 6px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.guide-card.safe {
    background: #eafaf6;
    border: 1px solid #00b894;
}
.guide-card.value {
    background: #eaf4ff;
    border: 1px solid #0984e3;
}
.guide-card.warning {
    background: #fff8ee;
    border: 1px solid #fdcb6e;
}
.guide-card .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 8px;
}
.guide-card.safe    .card-title { color: #00b894; }
.guide-card.value   .card-title { color: #0984e3; }
.guide-card.warning .card-title { color: #e17055; }
.guide-card .card-main {
    color: #2d3436;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.guide-card .card-sub {
    color: #636e72;
    font-size: 0.84rem;
    line-height: 1.5;
}

/* ===== TABS ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px 10px 0 0;
    gap: 4px;
    padding: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: #636e72;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 0.88rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #0d3349 !important;
    color: #ffffff !important;
}

/* ===== BOUTON ANALYSER ===== */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d3349 0%, #0984e3 100%);
    color: #ffffff;
    font-weight: 800;
    border: none;
    border-radius: 8px;
    font-size: 0.92rem;
    letter-spacing: 0.5px;
    padding: 10px 28px;
    transition: opacity 0.2s;
    box-shadow: 0 2px 8px rgba(9,132,227,0.3);
}
[data-testid="stButton"] > button:hover { opacity: 0.88; }

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 14px !important;
    border: 1px solid #dfe6e9;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { color: #636e72 !important; }
[data-testid="stMetricValue"] { color: #2d3436 !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { color: #00b894 !important; }

/* ===== DIVIDER ===== */
hr { border-color: #dfe6e9 !important; }
</style>
""", unsafe_allow_html=True)

# ---- Header principal ----
st.markdown("""
<div class="betpredict-header">
  <div class="betpredict-logo">⚽</div>
  <div>
    <h1>BetPredict Pro</h1>
    <div class="tagline">Analyse statistique &amp; aide à la décision</div>
  </div>
</div>
""", unsafe_allow_html=True)

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

