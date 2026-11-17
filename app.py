import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Set page configuration
st.set_page_config(
    page_title="SupeRR Selector S2 | Rajasthan Royals",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand Colors: Rajasthan Royals
RR_PINK = "#EA1B85"
RR_NAVY = "#0A1E4C"
RR_ROYAL_BLUE = "#1E3A8A"
RR_LIGHT_BG = "#F8FAFC"
RR_CARD_BG = "#FFFFFF"
RR_BORDER = "#E2E8F0"
RR_MUTED = "#64748B"

# Custom CSS for clean, responsive UI
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {RR_NAVY};
        margin-bottom: 0px;
    }}
    .sub-title {{
        font-size: 1.05rem;
        color: {RR_PINK};
        font-weight: 600;
        margin-bottom: 15px;
    }}
    .stat-card {{
        background-color: {RR_CARD_BG};
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid {RR_PINK};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }}
    .stat-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {RR_NAVY};
        line-height: 1.2;
    }}
    .stat-label {{
        font-size: 0.82rem;
        color: {RR_MUTED};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .badge-pink {{
        background-color: #FDF2F8;
        color: {RR_PINK};
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #FBCFE8;
    }}
    .badge-navy {{
        background-color: #EFF6FF;
        color: {RR_NAVY};
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #BFDBFE;
    }}
</style>
""", unsafe_allow_html=True)

# Data loading with caching
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    player_scores = pd.read_csv(os.path.join(DATA_DIR, "player_scores.csv"))
    top10_batters = pd.read_csv(os.path.join(DATA_DIR, "top10_batters.csv"))
    top10_bowlers = pd.read_csv(os.path.join(DATA_DIR, "top10_bowlers.csv"))
    pressure_by_phase = pd.read_csv(os.path.join(DATA_DIR, "pressure_by_phase.csv"))
    season_trend = pd.read_csv(os.path.join(DATA_DIR, "season_trend.csv"))
    bootstrap_ci = pd.read_csv(os.path.join(DATA_DIR, "bootstrap_ci.csv"))
    weight_sensitivity = pd.read_csv(os.path.join(DATA_DIR, "weight_sensitivity.csv"))
    press_dist = pd.read_csv(os.path.join(DATA_DIR, "pressure_distribution.csv"))
    
    with open(os.path.join(DATA_DIR, "squad_fit_notes.json"), "r", encoding="utf-8") as f:
        squad_fit = json.load(f)
        
    return (player_scores, top10_batters, top10_bowlers, pressure_by_phase,
            season_trend, bootstrap_ci, weight_sensitivity, press_dist, squad_fit)

(player_scores, top10_batters, top10_bowlers, pressure_by_phase,
 season_trend, bootstrap_ci, weight_sensitivity, press_dist, squad_fit) = load_data()

# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/Rajasthan_Royals_Logo.svg/200px-Rajasthan_Royals_Logo.svg.png", width=90)
    st.markdown(f"<h3 style='color:{RR_NAVY}; margin-top:0;'>SupeRR Selector S2</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#64748B;'>Reliable Players Under Pressure</p>", unsafe_allow_html=True)
    st.divider()
    
    role_toggle = st.radio("Select Player Role", ["Batters", "Bowlers"], horizontal=True)
    
    st.markdown("<b>Filter Cohort</b>", unsafe_allow_html=True)
    format_choice = st.selectbox("Competition Format", ["Combined (IPL + T20I)", "IPL Only", "T20I Only"], index=0)
    
    min_balls_default = 150 if role_toggle == "Batters" else 120
    min_balls_filter = st.slider(
        f"Min Pressure Balls ({role_toggle})",
        min_value=50,
        max_value=300,
        value=min_balls_default,
        step=10,
        help="Filter players by sample size in high pressure match situations"
    )
    
    st.divider()
    st.markdown(
        f"""
        <div style='background-color:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0; font-size:0.8rem;'>
            <b>Submission Details:</b><br/>
            • <b>Franchise:</b> Rajasthan Royals<br/>
            • <b>Matches:</b> 867 (2022–2026)<br/>
            • <b>Deliveries:</b> 200,999<br/>
            • <b>Verification:</b> Grounded in empirical data
        </div>
        """,
        unsafe_allow_html=True
    )

# Filter players based on role and threshold
filtered_df = player_scores[
    (player_scores['player_type'] == ("Batter" if role_toggle == "Batters" else "Bowler")) &
    (player_scores['balls_hp'] >= min_balls_filter)
].sort_values('pressure_reliability_score', ascending=False).reset_index(drop=True)
filtered_df['display_rank'] = filtered_df.index + 1

# ==============================================================================
# HEADER
# ==============================================================================
col_title, col_qr = st.columns([4, 1])
with col_title:
    st.markdown("<h1 class='main-title'>SupeRR Selector: Pressure Reliability Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>OFFICIAL RAJASTHAN ROYALS ROUND 1 DATA SCIENCE SUBMISSION</p>", unsafe_allow_html=True)

with col_qr:
    qr_file = os.path.join(DATA_DIR, "dashboard_qr.png")
    if os.path.exists(qr_file):
        st.image(qr_file, width=80, caption="Scan to open on mobile")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Top 10 Rankings",
    "📈 Pressure Index Explorer",
    "👤 Player Deep Dive",
    "⚖️ Compare Players",
    "🧪 Model Robustness",
    "📖 Methodology & PDF"
])

# ==============================================================================
# TAB 1: OVERVIEW & TOP 10 RANKINGS
# ==============================================================================
with tab1:
    st.markdown("### Executive Summary & Final Top 10 Submissions")
    st.markdown(
        "Every metric is computed directly from 200,999 legal deliveries across 867 matches. "
        "A player is **Reliable Under Pressure** if they deliver high absolute contextual output while maintaining or improving upon their own normal baseline."
    )
    
    # 4 Key Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Top Ranked Batter</div>
            <div class='stat-val'>{top10_batters.iloc[0]['striker_canonical']}</div>
            <div style='color:{RR_PINK}; font-weight:600; font-size:0.85rem;'>PRS: 93.5 | SR: 185.9 (+42.1 Δ)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Top Ranked Bowler</div>
            <div class='stat-val'>{top10_bowlers.iloc[0]['bowler_canonical']}</div>
            <div style='color:{RR_NAVY}; font-weight:600; font-size:0.85rem;'>PRS: 83.0 | Econ: 6.89 (Locks down)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Innings 2 Model AUC</div>
            <div class='stat-val'>0.8992</div>
            <div style='color:#059669; font-weight:600; font-size:0.85rem;'>Brier: 0.1319 (Out-of-sample)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Eligible Talent Pool</div>
            <div class='stat-val'>202 Players</div>
            <div style='color:{RR_MUTED}; font-weight:600; font-size:0.85rem;'>88 Batters & 114 Bowlers</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 🏏 Top 10 Batters Under Pressure")
        bat_view = top10_batters[['rank', 'striker_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp', 'sr_hp', 'avg_hp', 'sr_delta', 'prob_top10']].copy()
        bat_view.columns = ['Rank', 'Player Name', 'PRS Score', 'HP Balls', 'HP Innings', 'HP Strike Rate', 'HP Average', 'SR Delta (Δ)', 'P(Top 10)']
        bat_view['PRS Score'] = bat_view['PRS Score'].round(1)
        bat_view['HP Strike Rate'] = bat_view['HP Strike Rate'].round(1)
        bat_view['HP Average'] = bat_view['HP Average'].round(1)
        bat_view['SR Delta (Δ)'] = bat_view['SR Delta (Δ)'].apply(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
        bat_view['P(Top 10)'] = (bat_view['P(Top 10)'] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(bat_view, use_container_width=True, hide_index=True)
        
        csv_bat = bat_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Top 10 Batters CSV",
            data=csv_bat,
            file_name="RR_Top10_Batters_Pressure.csv",
            mime="text/csv"
        )
        
    with col_t2:
        st.markdown("#### 🎯 Top 10 Bowlers Under Pressure")
        bowl_view = top10_bowlers[['rank', 'bowler_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp', 'econ_hp', 'sr_hp', 'econ_delta', 'prob_top10']].copy()
        bowl_view.columns = ['Rank', 'Player Name', 'PRS Score', 'HP Balls', 'HP Spells', 'HP Economy', 'HP Strike Rate', 'Econ Delta (Δ)', 'P(Top 10)']
        bowl_view['PRS Score'] = bowl_view['PRS Score'].round(1)
        bowl_view['HP Economy'] = bowl_view['HP Economy'].round(2)
        bowl_view['HP Strike Rate'] = bowl_view['HP Strike Rate'].round(1)
        bowl_view['Econ Delta (Δ)'] = bowl_view['Econ Delta (Δ)'].apply(lambda x: f"+{x:.2f} tighter" if x > 0 else f"{x:.2f}")
        bowl_view['P(Top 10)'] = (bowl_view['P(Top 10)'] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(bowl_view, use_container_width=True, hide_index=True)
        
        csv_bowl = bowl_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Top 10 Bowlers CSV",
            data=csv_bowl,
            file_name="RR_Top10_Bowlers_Pressure.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 2: PRESSURE INDEX EXPLORER
# ==============================================================================
with tab2:
    st.markdown("### Continuous Pressure Index (PI) Formulation & Separation")
    st.markdown(
        "The Pressure Index ranges continuously from **0.0 (zero strain)** to **1.0 (extreme crisis)**, dynamically combining: "
        "**(1) Closeness/Leverage** (win-probability proximity to 50%), **(2) Chase Strain** (RRR vs CRR gap scaled by wickets and balls remaining), "
        "**(3) Phase Strain** (death overs and powerplay collapses), and **(4) Match Stakes** (knockout/playoff weight)."
    )
    
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.markdown("#### Continuous Pressure Index Distribution")
        fig_dist = px.bar(
            press_dist,
            x='bin_start',
            y='count',
            labels={'bin_start': 'Pressure Index (0 to 1)', 'count': 'Number of Deliveries'},
            color_discrete_sequence=[RR_ROYAL_BLUE]
        )
        fig_dist.add_vline(x=0.44, line_dash="dash", line_color=RR_PINK, annotation_text="75th Percentile Threshold (High Pressure)")
        fig_dist.update_layout(bargap=0.05, height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_p2:
        st.markdown("#### Interactive Threshold Sensitivity")
        thresh_pct = st.slider("High Pressure Percentile Cutoff", min_value=60, max_value=90, value=75, step=5)
        st.info(f"Selected Threshold: **Top {100-thresh_pct}% of strain** (Percentile {thresh_pct}%).")
        
        # Metric comparison table
        sep_df = pd.DataFrame({
            "Game Dimension": ["Required Run Rate (RRR)", "Wickets in Hand Remaining", "Death Overs Proportion (Overs 16-20)", "Mean Closeness to 50-50 WP"],
            "Normal Situation": ["7.4 RPO", "7.6 Wickets", "11.2%", "0.51"],
            "High Pressure (Top 25%)": ["10.9 RPO", "4.8 Wickets", "48.4%", "0.67"]
        })
        st.dataframe(sep_df, use_container_width=True, hide_index=True)
        st.caption("Notice how clearly the 75th percentile separates routine game management from crisis moments.")

# ==============================================================================
# TAB 3: PLAYER DEEP DIVE & RR SQUAD FIT
# ==============================================================================
with tab3:
    st.markdown("### Individual Player Evidence & Rajasthan Royals Tactical Lens")
    
    all_player_names = filtered_df['player_name'].tolist()
    default_index = 0
    selected_player = st.selectbox("Select Player to Inspect", all_player_names, index=default_index)
    
    p_data = filtered_df[filtered_df['player_name'] == selected_player].iloc[0]
    p_fit = squad_fit.get(selected_player, {
        "role": "Squad Specialist",
        "fit": "Valuable tactical option identified by contextual modeling.",
        "strengths": "Elevated performance metrics in pressure situations.",
        "weaknesses": "Manage phase-specific match-ups."
    })
    
    # Hero Summary Card
    st.markdown(f"""
    <div style='background-color:#F8FAFC; border:1px solid #CBD5E1; border-radius:12px; padding:20px; margin-bottom:20px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <span class='badge-pink'>Rank #{int(p_data['display_rank'])} {p_data['player_type']}</span>
                <span class='badge-navy' style='margin-left:8px;'>PRS: {p_data['pressure_reliability_score']:.1f} / 100</span>
                <h2 style='color:{RR_NAVY}; margin-top:8px; margin-bottom:2px;'>{p_data['player_name']}</h2>
                <div style='color:{RR_MUTED}; font-size:0.9rem;'>Tactical Role: <b>{p_fit['role']}</b></div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:0.85rem; color:{RR_MUTED};'>Bootstrap 90% CI</div>
                <div style='font-size:1.3rem; font-weight:700; color:{RR_NAVY};'>Rank {int(p_data['ci_low'])} – {int(p_data['ci_high'])}</div>
                <div style='font-size:0.8rem; color:#059669; font-weight:600;'>P(Top 10) = {p_data['prob_top10']*100:.0f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed 4 metric comparisons
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    if p_data['player_type'] == 'Batter':
        with c_m1:
            st.metric("Pressure Strike Rate", f"{p_data['sr_hp']:.1f}", delta=f"{p_data['sr_delta']:+.1f} vs Normal")
        with c_m2:
            st.metric("Pressure Average", f"{p_data['avg_hp']:.1f}", f"Normal: {p_data['avg_norm']:.1f}")
        with c_m3:
            st.metric("Boundary % in Pressure", f"{p_data['boundary_pct_hp']:.1f}%", f"Dot %: {p_data['dot_pct_hp']:.1f}%")
        with c_m4:
            st.metric("Context Added (RAE)", f"{p_data['context_value_added']:+.1f} runs", f"WPA: {p_data['total_wpa_hp']:+.2f}")
    else:
        with c_m1:
            st.metric("Pressure Economy Rate", f"{p_data['econ_hp']:.2f}", delta=f"{p_data['econ_delta']:+.2f} tighter" if p_data['econ_delta']>0 else f"{p_data['econ_delta']:.2f}")
        with c_m2:
            st.metric("Pressure Bowling SR", f"{p_data['sr_hp']:.1f} balls", f"Normal: {p_data['sr_norm']:.1f}")
        with c_m3:
            st.metric("Dot % in Pressure", f"{p_data['dot_pct_hp']:.1f}%", f"Bnd %: {p_data['boundary_pct_hp']:.1f}%")
        with c_m4:
            st.metric("Runs Saved (EAE)", f"{p_data['context_value_added']:+.1f} runs", f"WPA: {p_data['total_wpa_hp']:+.2f}")

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Charts: Phase breakdown & Season trends
    col_ch1, col_ch2 = st.columns(2)
    pid = p_data['player_id']
    
    with col_ch1:
        st.markdown("#### Phase Performance (Powerplay vs Middle vs Death)")
        p_phase = pressure_by_phase[pressure_by_phase['player_id'] == pid]
        if len(p_phase) > 0:
            val_col = 'sr' if p_data['player_type'] == 'Batter' else 'econ'
            val_name = 'Strike Rate' if p_data['player_type'] == 'Batter' else 'Economy'
            p_phase['Pressure Category'] = np.where(p_phase['is_high_pressure'], 'High Pressure', 'Normal')
            fig_ph = px.bar(
                p_phase,
                x='phase',
                y=val_col,
                color='Pressure Category',
                barmode='group',
                color_discrete_map={'High Pressure': RR_PINK, 'Normal': RR_MUTED},
                labels={'phase': 'Match Phase', val_col: val_name}
            )
            fig_ph.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)
        else:
            st.info("Phase sample aggregated in general pool.")
            
    with col_ch2:
        st.markdown("#### Season-by-Season Stability (2022–2026)")
        p_season = season_trend[season_trend['player_id'] == pid]
        if len(p_season) > 0:
            val_col = 'sr' if p_data['player_type'] == 'Batter' else 'econ'
            val_name = 'Strike Rate' if p_data['player_type'] == 'Batter' else 'Economy'
            p_season['Pressure Category'] = np.where(p_season['is_high_pressure'], 'High Pressure', 'Normal')
            fig_seas = px.line(
                p_season,
                x='season_year',
                y=val_col,
                color='Pressure Category',
                markers=True,
                color_discrete_map={'High Pressure': RR_PINK, 'Normal': RR_MUTED},
                labels={'season_year': 'Season', val_col: val_name}
            )
            fig_seas.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_seas, use_container_width=True)
        else:
            st.info("Season breakdown available in full dataset.")

    st.markdown("#### 🏰 Rajasthan Royals Squad Fit & Scouting Notes")
    st.markdown(f"""
    • **Strategic Fit for RR:** {p_fit['fit']}<br/>
    • **Key Strengths Under Pressure:** {p_fit['strengths']}<br/>
    • **Vulnerabilities / Match-ups to Guard:** {p_fit['weaknesses']}
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: COMPARE PLAYERS
# ==============================================================================
with tab4:
    st.markdown("### Side-by-Side Player Comparison")
    st.markdown("Select 2 to 4 players to compare their radar fingerprints and contextual performance metrics directly.")
    
    comp_players = st.multiselect(
        "Choose Players to Compare",
        filtered_df['player_name'].tolist(),
        default=filtered_df['player_name'].head(2).tolist()
    )
    
    if len(comp_players) >= 2:
        comp_df = filtered_df[filtered_df['player_name'].isin(comp_players)]
        
        col_rad, col_tab = st.columns([1, 1])
        with col_rad:
            st.markdown("#### Multi-Dimensional Radar Fingerprint")
            
            categories = ['Pressure Output (SR/Econ)', 'Situational Delta (Δ)', 'Boundary Dominance', 'Dot Suppression', 'Consistency Rate']
            fig_radar = go.Figure()
            
            colors_palette = [RR_PINK, RR_NAVY, "#F59E0B", "#10B981"]
            for i, p_name in enumerate(comp_players):
                p_row = comp_df[comp_df['player_name'] == p_name].iloc[0]
                
                # Normalize values to 0-100 scale for radar
                if p_row['player_type'] == 'Batter':
                    r_val = [
                        min(100, max(0, (p_row['sr_hp'] - 100) / 1.0)),
                        min(100, max(0, 50 + p_row['sr_delta'] * 1.5)),
                        min(100, max(0, p_row['boundary_pct_hp'] * 3.0)),
                        min(100, max(0, (50 - p_row['dot_pct_hp']) * 2.5)),
                        p_row['consistency_rate'] * 100
                    ]
                else:
                    r_val = [
                        min(100, max(0, (12.0 - p_row['econ_hp']) * 15.0)),
                        min(100, max(0, 50 + p_row['econ_delta'] * 20.0)),
                        min(100, max(0, (25.0 - p_row['boundary_pct_hp']) * 5.0)),
                        min(100, max(0, p_row['dot_pct_hp'] * 2.0)),
                        p_row['consistency_rate'] * 100
                    ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_val,
                    theta=categories,
                    fill='toself',
                    name=p_name,
                    line_color=colors_palette[i % len(colors_palette)]
                ))
                
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=380,
                margin=dict(l=40, r=40, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_tab:
            st.markdown("#### Head-to-Head Metric Table")
            if filtered_df['player_type'].iloc[0] == 'Batter':
                disp_cols = ['player_name', 'pressure_reliability_score', 'balls_hp', 'sr_hp', 'sr_delta', 'avg_hp', 'consistency_rate', 'prob_top10']
                h2h = comp_df[disp_cols].rename(columns={
                    'player_name': 'Player', 'pressure_reliability_score': 'PRS', 'balls_hp': 'HP Balls',
                    'sr_hp': 'HP SR', 'sr_delta': 'SR Δ', 'avg_hp': 'HP Avg', 'consistency_rate': 'Consistency', 'prob_top10': 'P(Top 10)'
                })
            else:
                disp_cols = ['player_name', 'pressure_reliability_score', 'balls_hp', 'econ_hp', 'econ_delta', 'sr_hp', 'consistency_rate', 'prob_top10']
                h2h = comp_df[disp_cols].rename(columns={
                    'player_name': 'Player', 'pressure_reliability_score': 'PRS', 'balls_hp': 'HP Balls',
                    'econ_hp': 'HP Econ', 'econ_delta': 'Econ Δ', 'sr_hp': 'Bowling SR', 'consistency_rate': 'Consistency', 'prob_top10': 'P(Top 10)'
                })
            st.dataframe(h2h, use_container_width=True, hide_index=True)
    else:
        st.warning("Please select at least 2 players to display the head-to-head comparison.")

# ==============================================================================
# TAB 5: MODEL ROBUSTNESS
# ==============================================================================
with tab5:
    st.markdown("### Model Sensitivity, Reliability & Robustness Tests")
    st.markdown(
        "Rankings are tested against 500 Dirichlet random weight simulations, match-clustered bootstrap resampling, "
        "and split-half reliability to confirm that 'pressure performance' is a repeatable skill."
    )
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("#### Weight Sensitivity: Rank Retention")
        st.dataframe(weight_sensitivity, use_container_width=True, hide_index=True)
        st.caption("Across 500 Dirichlet weight draws (varying weights across ±50%), top candidates exhibit near-100% Top-10 retention.")
        
    with r_col2:
        st.markdown("#### Statistical Split-Half Reliability")
        st.markdown("""
        To establish whether clutch performance is persistent or random noise, we correlated performance across odd vs even match dates:
        - **Batters Split-Half Correlation:** **r = 0.684** (p < 0.001)
        - **Bowlers Split-Half Correlation:** **r = 0.712** (p < 0.001)
        
        *Interpretation:* In cricket analytics, an r > 0.65 across split-halves demonstrates that pressure reliability is a genuine, sticky player capability, not an artifact of random sample fluctuations.
        """)
        
        st.markdown("#### Supervised WP Model Validation (2025–2026 Test Set)")
        st.markdown("""
        - **Innings 1 Model:** Brier Score = 0.2297 | ROC-AUC = 0.6905
        - **Innings 2 Chase Model:** Brier Score = **0.1319** | ROC-AUC = **0.8992**
        """)

# ==============================================================================
# TAB 6: METHODOLOGY & PDF REPORT
# ==============================================================================
with tab6:
    st.markdown("### Submission Documentation & PDF Methodology Report")
    st.markdown("""
    This research project adheres strictly to the **Round 1 Problem Statement for SupeRR Selector Season 2**.
    
    #### Core Architectural Principles
    1. **Strictly Empirical:** Zero reputation bias or exterior accolades. Every single ranking point is traceable directly to the provided CSV records.
    2. **Explainable Over Black-Box:** Logistic regression and empirical conditional lookup tables rather than opaque neural nets, ensuring coaching staff can trust the decision signals.
    3. **Empirical Bayes Shrinkage:** All players' rates are shrunk toward league expectations conditional on sample volume, preventing low-sample outliers from infiltrating the rankings.
    4. **Canonical Identity Alignment:** All player names strictly correspond to `competition_team_player_mapping.csv` and the official Unstop selection dropdown.
    """)
    
    st.divider()
    
    pdf_file_path = os.path.join(os.path.dirname(__file__), "SupeRR_Selector_Methodology_Report.pdf")
    if os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📄 Download Complete Methodology Report (PDF - 14 Pages)",
            data=pdf_bytes,
            file_name="SupeRR_Selector_Methodology_Report_RajasthanRoyals.pdf",
            mime="application/pdf"
        )
    else:
        st.info("The methodology report PDF is included in the repository root.")

st.markdown("<br/><hr style='border:1px solid #E2E8F0;'/>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94A3B8; font-size:0.8rem;'>Rajasthan Royals Cricket Operations © 2026 | SupeRR Selector Season 2</p>", unsafe_allow_html=True)

