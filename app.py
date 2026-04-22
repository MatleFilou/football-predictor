import csv
import os
from datetime import datetime

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
    "resultat_reel", "score_reel",
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
        "1": "1 — Victoire domicile",
        "N": "N — Match nul",
        "2": "2 — Victoire extérieur",
        "Over 2.5": "Over 2.5",
        "BTTS": "BTTS — Les 2 équipes marquent",
    }.get(code, code)

def render_lequipe_search(key_prefix: str, is_international: bool = False):
    """
    Charge les stats (V/N/D, buts) depuis lequipe.fr
    et le classement UEFA (clubs) ou FIFA (sélections) automatiquement.
    """
    col_name, col_btn = st.columns([3, 1])
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
        with st.spinner(f"Récupération des stats pour {name} sur lequipe.fr…"):
            try:
                from scraper_lequipe import scrape_team
                data = scrape_team(name)
                st.session_state[f"{key_prefix}_w"]  = int(data["wins"])
                st.session_state[f"{key_prefix}_d"]  = int(data["draws"])
                st.session_state[f"{key_prefix}_l"]  = int(data["losses"])
                st.session_state[f"{key_prefix}_gs"] = float(data["goals_scored_avg"])
                st.session_state[f"{key_prefix}_gc"] = float(data["goals_conceded_avg"])
                matches_preview = "\n".join(
                    f"• {m['date']}  {m['home_team']} {m['score_home']}-{m['score_away']} {m['away_team']}"
                    for m in data.get("matches", [])
                )
                st.success(
                    f"✅ **{data['team_name']}** — {data['nb_matches']} matchs (lequipe.fr)."
                    + rank_msg + "\n\n"
                    + matches_preview + "\n\n"
                    "Complétez buteur/passeur si besoin."
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

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        odd_home = st.number_input("Cote 1", min_value=1.01, value=2.00, key=f"{key_prefix}_odd_1")
    with c2:
        odd_draw = st.number_input("Cote N", min_value=1.01, value=3.20, key=f"{key_prefix}_odd_n")
    with c3:
        odd_away = st.number_input("Cote 2", min_value=1.01, value=2.00, key=f"{key_prefix}_odd_2")
    with c4:
        odd_over25 = st.number_input("Cote Over 2.5", min_value=1.01, value=1.90, key=f"{key_prefix}_odd_o25")
    with c5:
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

            best_financial = max([
                ("1", res["home_prob"], v_home),
                ("N", res["draw_prob"], v_draw),
                ("2", res["away_prob"], v_away),
                ("Over 2.5", goal_markets["over25"], v_over25),
                ("BTTS", goal_markets["btts"], v_btts),
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
                "Marché": ["1", "N", "2", "Over 2.5", "BTTS"],
                "Modèle (%)": [res["home_prob"], res["draw_prob"], res["away_prob"], goal_markets["over25"], goal_markets["btts"]],
                "Book (%)": [v_home[1], v_draw[1], v_away[1], v_over25[1], v_btts[1]],
                "Edge": [v_home[2], v_draw[2], v_away[2], v_over25[2], v_btts[2]],
                "Signal": [v_home[0], v_draw[0], v_away[0], v_over25[0], v_btts[0]],
            })

            st.markdown("---")
            st.subheader("🎯 Synthèse")
            st.write(f"**Résultat le plus probable :** {res['prediction']} — {res['label']}")
            st.write(f"**Pari le plus intéressant :** {format_market_label(best_financial[0])}")
            st.write(f"**Verdict global :** {reco}")
            st.write(f"**Indice de confiance :** {index}")

            rang_h = home.get("uefa_rank", home.get("fifa_rank", 0))
            rang_a = away.get("uefa_rank", away.get("fifa_rank", 0))

            log_analyse({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "competition": competition_label,
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
                "resultat_reel": "", "score_reel": "",
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

st.title("⚽ Mode Parieur")

tab1, tab2 = st.tabs(["🏟️ Clubs", "🌍 Sélections internationales"])

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

