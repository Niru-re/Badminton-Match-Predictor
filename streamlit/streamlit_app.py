"""BWF Badminton Match Predictor — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import streamlit as st

from src.predict import DISCIPLINES, list_players, predict_match, compute_live_features
from src.config import MODELS_DIR, PROCESSED_DIR

# ── page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="BWF Match Predictor",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0f0d; color: #e8f5e9; }
[data-testid="stSidebar"]          { background: #0d1810; }
.metric-card {
    background: #1a2e1a; border: 1px solid #2d6a4f;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.winner-card {
    background: #1a3a1a; border: 2px solid #f4d03f;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.stProgress > div > div { background-color: #f4d03f !important; }
h1, h2, h3 { color: #f4d03f !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────
DISC_LABELS = {
    "MS": "Men's Singles", "WS": "Women's Singles",
    "MD": "Men's Doubles", "WD": "Women's Doubles", "XD": "Mixed Doubles",
}
TIERS  = [100, 300, 500, 750, 1000]
ROUNDS = {"Early (32)": 1, "Round of 16": 2, "Quarter-final": 3, "Semi-final": 4, "Final": 5}

def tc(s: str) -> str:
    return s.replace("/", " / ").title()

@st.cache_data
def load_metrics(discipline: str) -> dict:
    p = MODELS_DIR / f"metrics_{discipline.lower()}.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_features_df(discipline: str) -> pd.DataFrame:
    p = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    return pd.read_csv(p, parse_dates=["match_date"]) if p.exists() else pd.DataFrame()

@st.cache_data
def load_matches_df() -> pd.DataFrame:
    p = PROCESSED_DIR / "matches_clean.csv"
    return pd.read_csv(p, parse_dates=["match_date"]) if p.exists() else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏸 BWF Match Predictor")
    st.markdown("*ML-powered predictions using player strength, form, H2H & fatigue.*")
    st.divider()

    tab_choice = st.radio("Navigate", ["⚡ Predictor", "📊 Dashboard", "📈 Analytics"], label_visibility="collapsed")
    st.divider()

    discipline = st.selectbox("Discipline", DISCIPLINES, format_func=lambda d: DISC_LABELS[d])
    tier_label = st.selectbox("Tournament Tier", [f"Super {t}" for t in TIERS])
    tier = int(tier_label.split()[1])
    round_label = st.selectbox("Round", list(ROUNDS.keys()), index=2)
    round_imp = ROUNDS[round_label]

    players = list_players(discipline)
    st.divider()
    st.caption(f"🗃️ {len(players)} players with match history")

    metrics = load_metrics(discipline)
    if metrics:
        st.caption(f"🤖 Model: **{metrics.get('best_model','?')}**")
        st.caption(f"🎯 Accuracy: **{metrics.get('best_accuracy',0)*100:.1f}%**")

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════
if tab_choice == "⚡ Predictor":
    st.title("🏸 BWF Match Predictor")
    st.markdown(f"**{DISC_LABELS[discipline]}** · {tier_label} · {round_label}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player / Team 1", players, format_func=tc, key="p1")
    with col2:
        p2_opts = [p for p in players if p != p1]
        p2 = st.selectbox("Player / Team 2", p2_opts, format_func=tc, key="p2")

    st.divider()

    if st.button("⚡ Predict Winner", use_container_width=True, type="primary"):
        with st.spinner("Running ML inference…"):
            try:
                result = predict_match(p1, p2, discipline, tier, round_imp)

                # ── winner banner ──
                winner = result["predicted_winner"]
                st.markdown(f"""
                <div class="winner-card">
                    <p style="color:#f4d03f;font-size:0.85rem;letter-spacing:2px;margin:0">PREDICTED WINNER</p>
                    <h2 style="margin:0.3rem 0">{tc(winner)}</h2>
                </div>""", unsafe_allow_html=True)
                st.markdown("")

                # ── probability bars ──
                c1, c2 = st.columns(2)
                p1_prob = result["player1_win_probability"]
                p2_prob = result["player2_win_probability"]

                with c1:
                    is_w = winner == p1
                    st.markdown(f"**{'🏆 ' if is_w else ''}{tc(p1)}**")
                    st.progress(p1_prob)
                    st.metric("Win Probability", f"{p1_prob*100:.1f}%")

                with c2:
                    is_w = winner == p2
                    st.markdown(f"**{'🏆 ' if is_w else ''}{tc(p2)}**")
                    st.progress(p2_prob)
                    st.metric("Win Probability", f"{p2_prob*100:.1f}%")

                # ── feature breakdown ──
                st.divider()
                st.markdown("#### Feature Breakdown")
                feats = result["features"]
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Strength Diff",  f"{feats['strength_diff']:+.3f}")
                fc2.metric("H2H Diff",       f"{feats['head_to_head_diff']:+d} wins")
                fc3.metric("Form P1",        f"{feats['recent_form_p1']*100:.0f}%")
                fc4.metric("Form P2",        f"{feats['recent_form_p2']*100:.0f}%")

                fc5, fc6, fc7, fc8 = st.columns(4)
                fc5.metric("Fatigue P1",     f"{feats['fatigue_p1']} matches/wk")
                fc6.metric("Fatigue P2",     f"{feats['fatigue_p2']} matches/wk")
                fc7.metric("Tier",           f"Super {feats['tournament_tier']}")
                fc8.metric("Round Weight",   feats['round_importance'])

            except Exception as e:
                st.error(str(e))

# ══════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD (court + live scorer)
# ══════════════════════════════════════════════════════════════════
elif tab_choice == "📊 Dashboard":
    st.title("📊 Match Dashboard")
    st.markdown(f"**{DISC_LABELS[discipline]}** · {tier_label} · {round_label}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player / Team 1", players, format_func=tc, key="d_p1")
    with col2:
        p2_opts = [p for p in players if p != p1]
        p2 = st.selectbox("Player / Team 2", p2_opts, format_func=tc, key="d_p2")

    st.divider()

    # ── live scoreboard ──────────────────────────────────────────
    st.markdown("#### 🏸 Live Scoreboard")
    if "sets" not in st.session_state:
        st.session_state.sets = [[0, 0]]
        st.session_state.cur_set = 0
        st.session_state.rally_log = []

    cur = st.session_state.cur_set
    sets = st.session_state.sets

    # Score display
    score_cols = st.columns(len(sets) + 3)
    score_cols[0].markdown(f"**{tc(p1)[:20]}**")
    for i, s in enumerate(sets):
        color = "#f4d03f" if i == cur else "#aaa"
        score_cols[i+1].markdown(f"<span style='color:{color};font-size:1.5rem;font-weight:bold'>{s[0]}</span>", unsafe_allow_html=True)

    score_cols[0].markdown(f"**{tc(p2)[:20]}**")
    for i, s in enumerate(sets):
        color = "#f4d03f" if i == cur else "#aaa"
        score_cols[i+1].markdown(f"<span style='color:{color};font-size:1.5rem;font-weight:bold'>{s[1]}</span>", unsafe_allow_html=True)

    # Point buttons
    bc1, bc2, bc3, bc4 = st.columns([2, 2, 2, 2])
    if bc1.button(f"➕ Point → {tc(p1)[:15]}", use_container_width=True):
        st.session_state.sets[cur][0] += 1
        st.rerun()
    if bc2.button(f"➕ Point → {tc(p2)[:15]}", use_container_width=True):
        st.session_state.sets[cur][1] += 1
        st.rerun()
    if bc3.button("🆕 New Set", use_container_width=True) and cur < 2:
        st.session_state.sets.append([0, 0])
        st.session_state.cur_set += 1
        st.rerun()
    if bc4.button("🔄 Reset", use_container_width=True):
        st.session_state.sets = [[0, 0]]
        st.session_state.cur_set = 0
        st.session_state.rally_log = []
        st.rerun()

    st.divider()

    # ── court diagram ─────────────────────────────────────────────
    st.markdown("#### 🟩 Court View")
    is_doubles = discipline in ["MD", "WD", "XD"]
    p1s = p1.split()[0].upper() if p1 else "P1"
    p2s = p2.split()[0].upper() if p2 else "P2"

    court_svg = f"""
    <svg viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:10px">
      <rect width="600" height="280" fill="#1a7a3a"/>
      <rect x="40" y="20" width="520" height="240" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="2.5"/>
      {'<line x1="74" y1="20" x2="74" y2="260" stroke="rgba(255,255,255,0.4)" stroke-width="1.5"/><line x1="526" y1="20" x2="526" y2="260" stroke="rgba(255,255,255,0.4)" stroke-width="1.5"/>' if is_doubles else ''}
      <line x1="40" y1="140" x2="560" y2="140" stroke="white" stroke-width="3"/>
      <circle cx="300" cy="140" r="5" fill="white"/>
      <line x1="40" y1="100" x2="560" y2="100" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
      <line x1="40" y1="180" x2="560" y2="180" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
      <line x1="300" y1="20"  x2="300" y2="100" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
      <line x1="300" y1="180" x2="300" y2="260" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
      <circle cx="160" cy="185" r="22" fill="#f4d03f" opacity="0.9"/>
      <text x="160" y="191" text-anchor="middle" fill="#0d1810" font-size="10" font-weight="bold">{p1s}</text>
      <circle cx="440" cy="95" r="22" fill="#60a5fa" opacity="0.9"/>
      <text x="440" y="101" text-anchor="middle" fill="#0d1810" font-size="10" font-weight="bold">{p2s}</text>
      <text x="300" y="155" text-anchor="middle" fill="rgba(255,255,255,0.15)" font-size="14" font-weight="bold" letter-spacing="3">NET</text>
    </svg>"""
    st.markdown(court_svg, unsafe_allow_html=True)

    st.divider()

    # ── shot tagger ───────────────────────────────────────────────
    st.markdown("#### 🎯 Shot Tagger")
    tc1, tc2 = st.columns(2)
    shot_player = tc1.radio("Who hit?", [tc(p1)[:20], tc(p2)[:20]], horizontal=True)
    shot_outcome = tc2.radio("Outcome", ["In Play", "Winner", "Forced Error", "Unforced Error"], horizontal=True)

    sc1, sc2, sc3 = st.columns(3)
    shot_type  = sc1.selectbox("Shot Type", ["Forehand", "Backhand"])
    shot_phase = sc2.selectbox("Phase", ["Serve", "Return", "Attack", "Neutral", "Defense"])
    shot_var   = sc3.selectbox("Variation", ["Short", "Flick", "Drive", "Net", "Lift", "Smash"])

    if st.button("📝 Log Shot", use_container_width=True):
        cur_s = st.session_state.sets[st.session_state.cur_set]
        entry = {
            "Player": shot_player, "Type": shot_type, "Phase": shot_phase,
            "Variation": shot_var, "Outcome": shot_outcome,
            "Score": f"{cur_s[0]}-{cur_s[1]}",
        }
        st.session_state.rally_log.insert(0, entry)
        if shot_outcome in ["Winner", "Forced Error", "Unforced Error"]:
            if shot_outcome == "Winner":
                if shot_player == tc(p1)[:20]: st.session_state.sets[st.session_state.cur_set][0] += 1
                else: st.session_state.sets[st.session_state.cur_set][1] += 1
            else:
                if shot_player == tc(p1)[:20]: st.session_state.sets[st.session_state.cur_set][1] += 1
                else: st.session_state.sets[st.session_state.cur_set][0] += 1
        st.rerun()

    if st.session_state.rally_log:
        st.markdown("#### 📋 Rally Log")
        st.dataframe(pd.DataFrame(st.session_state.rally_log[:15]), use_container_width=True, hide_index=True)

    # ── prediction panel ──────────────────────────────────────────
    st.divider()
    if st.button("⚡ Get ML Prediction", use_container_width=True, type="primary"):
        with st.spinner("Predicting…"):
            try:
                result = predict_match(p1, p2, discipline, tier, round_imp)
                w = result["predicted_winner"]
                p1p = result["player1_win_probability"]
                p2p = result["player2_win_probability"]
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Predicted Winner", tc(w))
                mc2.metric(f"{tc(p1)[:15]} Win %", f"{p1p*100:.1f}%")
                mc3.metric(f"{tc(p2)[:15]} Win %", f"{p2p*100:.1f}%")
            except Exception as e:
                st.error(str(e))

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Analytics":
    st.title("📈 Analytics")
    st.markdown(f"**{DISC_LABELS[discipline]}** — dataset insights & model performance")
    st.divider()

    df_matches = load_matches_df()
    df_feats   = load_features_df(discipline)
    metrics    = load_metrics(discipline)

    if df_matches.empty:
        st.warning("No match data found.")
        st.stop()

    disc_df = df_matches[df_matches["discipline"] == discipline]

    # ── overview metrics ──
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Matches",    f"{len(disc_df):,}")
    mc2.metric("Unique Players",   len(set(disc_df["player1"]) | set(disc_df["player2"])))
    mc3.metric("Date Range",       f"{disc_df['match_date'].min().year}–{disc_df['match_date'].max().year}")
    mc4.metric("Model Accuracy",   f"{metrics.get('best_accuracy',0)*100:.1f}%" if metrics else "N/A")

    st.divider()
    ac1, ac2 = st.columns(2)

    # ── matches per year ──
    with ac1:
        st.markdown("#### Matches per Year")
        yearly = disc_df.copy()
        yearly["year"] = yearly["match_date"].dt.year
        chart = yearly.groupby("year").size().reset_index(name="matches")
        st.bar_chart(chart.set_index("year"))

    # ── model comparison ──
    with ac2:
        st.markdown("#### Model Accuracy Comparison")
        if metrics and "results" in metrics:
            model_data = {k: v["accuracy"] for k, v in metrics["results"].items()}
            mdf = pd.DataFrame(model_data.items(), columns=["Model", "Accuracy"])
            mdf["Accuracy %"] = (mdf["Accuracy"] * 100).round(2)
            st.bar_chart(mdf.set_index("Model")["Accuracy"])
        else:
            st.info("No model metrics available.")

    st.divider()
    bc1, bc2 = st.columns(2)

    # ── top players by wins ──
    with bc1:
        st.markdown("#### Top 10 Players by Wins")
        if not disc_df.empty:
            p1w = disc_df[disc_df["player1_wins"] == 1]["player1"]
            p2w = disc_df[disc_df["player1_wins"] == 0]["player2"]
            all_wins = pd.concat([p1w, p2w]).value_counts().head(10).reset_index()
            all_wins.columns = ["Player", "Wins"]
            all_wins["Player"] = all_wins["Player"].apply(tc)
            st.dataframe(all_wins, use_container_width=True, hide_index=True)

    # ── feature distributions ──
    with bc2:
        st.markdown("#### Feature Distribution")
        if not df_feats.empty:
            feat_choice = st.selectbox("Feature", ["strength_diff", "recent_form_p1", "head_to_head_diff", "fatigue_p1"])
            st.bar_chart(df_feats[feat_choice].value_counts().sort_index().head(30))
        else:
            st.info("No feature data.")

    st.divider()

    # ── H2H lookup ──
    st.markdown("#### 🔍 Head-to-Head Lookup")
    hc1, hc2 = st.columns(2)
    hq1 = hc1.selectbox("Player A", players, format_func=tc, key="h2h_p1")
    hq2 = hc2.selectbox("Player B", [p for p in players if p != hq1], format_func=tc, key="h2h_p2")

    h2h_df = disc_df[
        ((disc_df["player1"] == hq1) & (disc_df["player2"] == hq2)) |
        ((disc_df["player1"] == hq2) & (disc_df["player2"] == hq1))
    ].sort_values("match_date", ascending=False)

    if h2h_df.empty:
        st.info("No head-to-head matches found.")
    else:
        p1w = int(((h2h_df["player1"] == hq1) & (h2h_df["player1_wins"] == 1)).sum() +
                  ((h2h_df["player2"] == hq1) & (h2h_df["player1_wins"] == 0)).sum())
        p2w = len(h2h_df) - p1w
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric(f"{tc(hq1)[:18]} Wins", p1w)
        hm2.metric("Total Matches", len(h2h_df))
        hm3.metric(f"{tc(hq2)[:18]} Wins", p2w)
        st.dataframe(
            h2h_df[["match_date", "tournament", "round", "player1", "player2", "player1_wins"]]
            .rename(columns={"player1_wins": "p1_won"}).head(10),
            use_container_width=True, hide_index=True,
        )
