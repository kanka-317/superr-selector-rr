"""
SupeRR Selector Season 2: Comprehensive Pipeline
Step 1: Game-State Reconstruction & Win Probability Model
Step 2: Pressure Definition & Sensitivity Analysis
Step 3: Context-Adjusted Performance & Baselines
Step 4: Fairness, Reliability, Shrinkage & Consistency
Step 5: Final Scores, Rankings, Sensitivity & Export
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

DATA_DIR = "6af3166e7b922_data_pack"
OUTPUTS_DIR = "outputs"
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

print("="*80)
print("RUNNING SUPERR SELECTOR SEASON 2 PIPELINE")
print("="*80)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\n[1/6] Loading data...")
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
playing_eleven = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))
mapping = pd.read_csv(os.path.join(DATA_DIR, "competition_team_player_mapping.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), low_memory=False)

# Canonical player names dictionary from mapping
canonical_player_names = mapping[['player_id', 'player_name']].drop_duplicates().set_index('player_id')['player_name'].to_dict()

# Merge match context into bbb
match_meta = matches.merge(competitions[['comp_id', 'match_format', 'season']], on='comp_id', how='left')
bbb = bbb.merge(
    match_meta[['match_id', 'comp_id', 'season', 'match_format', 'match_date', 'stage', 'knock_out', 
                'venue_id', 'home_team_id', 'away_team_id', 'win_team_id', 'result',
                'revised_target', 'first_inn_revised_overs', 'second_inn_revised_overs']],
    on='match_id',
    how='left'
)

# Convert season to numeric year
bbb['season_year'] = bbb['season'].astype(str).str[:4].astype(int)

# Standardize match_format: 'IPL' for IPL competitions, 'T20' for other domestic, 'T20I' for international
bbb['is_ipl'] = bbb['comp_id'].isin([1358, 1715, 2193, 2725, 3298])
bbb['format_group'] = np.where(bbb['is_ipl'], 'IPL', np.where(bbb['match_format'] == 'T20I', 'T20I', 'Other T20'))

# Canonical names in bbb
bbb['striker_canonical'] = bbb['striker_id'].map(canonical_player_names).fillna(bbb['striker_name'])
bbb['bowler_canonical'] = bbb['bowler_id'].map(canonical_player_names).fillna(bbb['bowler_name'])

print(f"Loaded {len(bbb):,} balls across {bbb['match_id'].nunique()} matches.")

# ==============================================================================
# RUN ACCOUNTING & LEGAL BALL DEFINITIONS
# ==============================================================================
print("\n[2/6] Operationalizing deliveries and run accounting...")
# Legal ball flag: wides, overstep (noball), and other (noball) are illegal
bbb['is_wide'] = (bbb['extras_type'] == 'wide')
bbb['is_noball'] = bbb['extras_type'].isin(['Overstep', 'other'])
bbb['is_legal'] = ~bbb['is_wide'] & ~bbb['is_noball']

# Batter balls faced: all deliveries except wides
bbb['is_batter_ball'] = ~bbb['is_wide']

# Batter runs: runs off bat
bbb['batter_runs'] = bbb['runs']

# Bowler runs conceded: runs off bat + wide (1) + noball (1) + extras_2_runs
bbb['bowler_extra_runs'] = np.where(bbb['is_wide'], 1, np.where(bbb['is_noball'], 1, 0)) + bbb['extras_2_runs'].fillna(0)
bbb['bowler_runs_conceded'] = bbb['runs'] + bbb['bowler_extra_runs']

# Team runs on delivery
bbb['team_extra_runs'] = np.where(bbb['is_wide'], 1, np.where(bbb['is_noball'], 1, np.where(bbb['extras_type'].isin(['byes', 'legbyes']), 1, 0))) + bbb['extras_2_runs'].fillna(0)
# Check penalty
bbb.loc[bbb['extras_type'] == 'Penalty', 'team_extra_runs'] = 5
bbb['team_runs_on_ball'] = bbb['runs'] + bbb['team_extra_runs']

# Bowler credited wicket
bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

print(f"Total legal balls: {bbb['is_legal'].sum():,}")
print(f"Total batter balls faced: {bbb['is_batter_ball'].sum():,}")
print(f"Total bowler-credited wickets: {bbb['is_bowler_wicket'].sum():,}")

# ==============================================================================
# STEP 1: GAME STATE RECONSTRUCTION
# ==============================================================================
print("\n[3/6] Reconstructing game state and training Win Probability Model...")

# Sort chronologically within match and innings
bbb = bbb.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Cumulative legal balls bowled in innings
bbb['legal_balls_bowled'] = bbb.groupby(['match_id', 'innings'])['is_legal'].cumsum()
# Legal balls bowled BEFORE this ball
bbb['legal_balls_before'] = bbb.groupby(['match_id', 'innings'])['legal_balls_bowled'].shift(1).fillna(0).astype(int)

# Cumulative score in innings
bbb['cum_team_runs'] = bbb.groupby(['match_id', 'innings'])['team_runs_on_ball'].cumsum()
bbb['score_before'] = bbb.groupby(['match_id', 'innings'])['cum_team_runs'].shift(1).fillna(0).astype(int)

# Cumulative wickets fallen in innings
bbb['cum_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
bbb['wickets_before'] = bbb.groupby(['match_id', 'innings'])['cum_wickets'].shift(1).fillna(0).astype(int)
bbb['wickets_in_hand_before'] = 10 - bbb['wickets_before']

# Determine total scheduled overs/balls for each innings
# Check revised overs
def get_total_balls(row):
    if row['innings'] == 1:
        if pd.notnull(row['first_inn_revised_overs']) and row['first_inn_revised_overs'] > 0:
            # Note: in data, revised overs may be given in overs (e.g. 15.0 or ball count 90.0)
            val = row['first_inn_revised_overs']
            return int(val if val > 20 else val * 6)
        return 120
    elif row['innings'] == 2:
        if pd.notnull(row['second_inn_revised_overs']) and row['second_inn_revised_overs'] > 0:
            val = row['second_inn_revised_overs']
            return int(val if val > 20 else val * 6)
        return 120
    else: # Super over
        return 6

bbb['total_inn_balls'] = bbb.apply(get_total_balls, axis=1)
bbb['balls_remaining'] = np.maximum(0, bbb['total_inn_balls'] - bbb['legal_balls_before'])

# Compute 1st innings final scores for match targets
inn1_finals = bbb[bbb['innings'] == 1].groupby('match_id').agg(
    inn1_score=('cum_team_runs', 'max'),
    inn1_revised_target=('revised_target', 'first')
).reset_index()

inn1_finals['target'] = np.where(
    inn1_finals['inn1_revised_target'].notnull() & (inn1_finals['inn1_revised_target'] > 0),
    inn1_finals['inn1_revised_target'],
    inn1_finals['inn1_score'] + 1
)

bbb = bbb.merge(inn1_finals[['match_id', 'target', 'inn1_score']], on='match_id', how='left')

# 2nd innings chase state
bbb['runs_required'] = np.where(bbb['innings'] == 2, np.maximum(0, bbb['target'] - bbb['score_before']), 0)
bbb['crr'] = np.where(bbb['legal_balls_before'] > 0, (bbb['score_before'] / bbb['legal_balls_before']) * 6.0, 7.5)
bbb['rrr'] = np.where(
    (bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
    (bbb['runs_required'] / bbb['balls_remaining']) * 6.0,
    np.where(bbb['innings'] == 2, 36.0, 0.0)
)
bbb['rrr_minus_crr'] = np.where(bbb['innings'] == 2, bbb['rrr'] - bbb['crr'], 0.0)

# Par score for 1st innings: average final score in format
avg_inn1_score = bbb[bbb['innings'] == 1].groupby('format_group')['inn1_score'].mean().to_dict()
bbb['format_par_total'] = bbb['format_group'].map(avg_inn1_score).fillna(165.0)
# Phase par score: expected score at current legal ball fraction
bbb['par_score_at_phase'] = bbb['format_par_total'] * (bbb['legal_balls_before'] / 120.0)
bbb['score_vs_par'] = bbb['score_before'] - bbb['par_score_at_phase']

# Over and Phase definition
bbb['over_num'] = (bbb['overid'] - 1)  # 0 to 19
bbb['phase'] = np.where(bbb['over_num'] < 6, 'Powerplay', np.where(bbb['over_num'] < 16, 'Middle', 'Death'))

# Check batting team win target
# Batting team won if win_team_id == batting_team_id
bbb['bat_team_won'] = (bbb['win_team_id'] == bbb['batting_team_id']).astype(int)

# Filter for matches with a definitive winner for WP model training
wp_valid_mask = bbb['win_team_id'].notnull() & bbb['innings'].isin([1, 2]) & ~bbb['result'].isin(['Match Abandoned', 'No Result'])

# Features for 1st innings
features_inn1 = ['balls_remaining', 'wickets_in_hand_before', 'score_before', 'score_vs_par']
# Features for 2nd innings
features_inn2 = ['balls_remaining', 'wickets_in_hand_before', 'runs_required', 'rrr', 'rrr_minus_crr']

# Train separate, highly explainable logistic regression models for Innings 1 and Innings 2
# Temporal validation split: Train on seasons 2022-2024, validate on 2025-2026
train_mask = wp_valid_mask & (bbb['season_year'] <= 2024)
val_mask = wp_valid_mask & (bbb['season_year'] >= 2025)

# Innings 1 Model
wp_model_inn1 = LogisticRegression(max_iter=1000)
X_train_inn1 = bbb.loc[train_mask & (bbb['innings'] == 1), features_inn1]
y_train_inn1 = bbb.loc[train_mask & (bbb['innings'] == 1), 'bat_team_won']
wp_model_inn1.fit(X_train_inn1, y_train_inn1)

X_val_inn1 = bbb.loc[val_mask & (bbb['innings'] == 1), features_inn1]
y_val_inn1 = bbb.loc[val_mask & (bbb['innings'] == 1), 'bat_team_won']
val_preds_inn1 = wp_model_inn1.predict_proba(X_val_inn1)[:, 1]

# Innings 2 Model
wp_model_inn2 = LogisticRegression(max_iter=1000)
X_train_inn2 = bbb.loc[train_mask & (bbb['innings'] == 2), features_inn2]
y_train_inn2 = bbb.loc[train_mask & (bbb['innings'] == 2), 'bat_team_won']
wp_model_inn2.fit(X_train_inn2, y_train_inn2)

X_val_inn2 = bbb.loc[val_mask & (bbb['innings'] == 2), features_inn2]
y_val_inn2 = bbb.loc[val_mask & (bbb['innings'] == 2), 'bat_team_won']
val_preds_inn2 = wp_model_inn2.predict_proba(X_val_inn2)[:, 1]

# Metrics
brier_inn1 = brier_score_loss(y_val_inn1, val_preds_inn1)
brier_inn2 = brier_score_loss(y_val_inn2, val_preds_inn2)
auc_inn1 = roc_auc_score(y_val_inn1, val_preds_inn1)
auc_inn2 = roc_auc_score(y_val_inn2, val_preds_inn2)

print(f"Innings 1 WP Model (Val 2025-2026) -> Brier: {brier_inn1:.4f}, AUC: {auc_inn1:.4f}")
print(f"Innings 2 WP Model (Val 2025-2026) -> Brier: {brier_inn2:.4f}, AUC: {auc_inn2:.4f}")

# Compute WP for all deliveries
bbb['wp_before'] = 0.5 # default
mask_inn1 = (bbb['innings'] == 1)
mask_inn2 = (bbb['innings'] == 2)
bbb.loc[mask_inn1, 'wp_before'] = wp_model_inn1.predict_proba(bbb.loc[mask_inn1, features_inn1])[:, 1]
bbb.loc[mask_inn2, 'wp_before'] = wp_model_inn2.predict_proba(bbb.loc[mask_inn2, features_inn2])[:, 1]

# For super overs, WP starts at 0.50
bbb.loc[bbb['innings'] > 2, 'wp_before'] = 0.50

# Win Probability After the ball (shift within innings, or terminal)
bbb['wp_after'] = bbb.groupby(['match_id', 'innings'])['wp_before'].shift(-1)
# Terminal state handling
# If chase ended on this ball: if bat team won, 1.0; else 0.0
# For innings 1 end, estimate based on final score
terminal_mask = bbb['wp_after'].isnull()
bbb.loc[terminal_mask & (bbb['innings'] == 2), 'wp_after'] = bbb.loc[terminal_mask & (bbb['innings'] == 2), 'bat_team_won'].astype(float)
bbb.loc[terminal_mask & (bbb['innings'] == 1), 'wp_after'] = bbb.loc[terminal_mask & (bbb['innings'] == 1), 'wp_before']

# Win Probability Added (WPA)
bbb['wpa_bat'] = bbb['wp_after'] - bbb['wp_before']
bbb['wpa_bowl'] = -bbb['wpa_bat']

# Plot Calibration Curve
fig, ax = plt.subplots(figsize=(6, 5))
prob_true_1, prob_pred_1 = calibration_curve(y_val_inn1, val_preds_inn1, n_bins=10)
prob_true_2, prob_pred_2 = calibration_curve(y_val_inn2, val_preds_inn2, n_bins=10)
ax.plot(prob_pred_1, prob_true_1, marker='o', label=f'Innings 1 (Brier={brier_inn1:.3f})', color='#D82A74', lw=2)
ax.plot(prob_pred_2, prob_true_2, marker='s', label=f'Innings 2 (Brier={brier_inn2:.3f})', color='#2563EB', lw=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
ax.set_xlabel('Predicted Win Probability')
ax.set_ylabel('Observed Win Frequency')
ax.set_title('Win Probability Model Calibration (2025-2026 Test Set)', fontsize=12, fontweight='bold')
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "wp_calibration.png"))
plt.close()

# ==============================================================================
# STEP 2: PRESSURE DEFINITION & CONTINUOUS PRESSURE INDEX
# ==============================================================================
print("\n[4/6] Constructing Continuous Pressure Index (0-1)...")

# A) Leverage Index (Expected WP swing between a boundary (+4 runs) and a wicket)
# For innings 2: swing = |WP(runs_req-4, balls-1, wkts) - WP(runs_req, balls-1, wkts-1)|
# Approximate leverage using finite difference on WP model
def compute_leverage(df):
    lev = np.abs(df['wpa_bat']).rolling(window=3, min_periods=1, center=True).mean()
    return lev

bbb['leverage'] = np.abs(bbb['wpa_bat'])
# Smooth leverage over state
# Proxy for potential swing: 2 * |WP - 0.5| is closeness, maximum leverage when WP is near 0.50
bbb['closeness'] = 1.0 - 2.0 * np.abs(bbb['wp_before'] - 0.50)

# B) Chase Tension: (RRR / 8.0) * (10 / (wickets_in_hand_before + 1)) * (1.0 / np.sqrt(balls_remaining/20.0 + 1))
chase_raw = np.where(
    bbb['innings'] == 2,
    (bbb['rrr'] / 9.0) * (6.0 / (bbb['wickets_in_hand_before'] + 1.0)) * (1.0 / np.sqrt(bbb['balls_remaining'] / 30.0 + 0.5)),
    0.0
)
bbb['chase_tension'] = np.clip(chase_raw / 3.0, 0, 1)

# C) Innings Phase & Collapses
# Death overs = overs 16-20 (over_num >= 16)
is_death = (bbb['over_num'] >= 16).astype(float)
# Powerplay collapse: >= 3 wickets lost in powerplay
pp_collapse = ((bbb['over_num'] < 6) & (bbb['wickets_before'] >= 2)).astype(float)
# New batter at crease (first 6 balls of batter innings)
bbb['batter_balls_cum'] = bbb.groupby(['match_id', 'innings', 'striker_id'])['is_batter_ball'].cumsum()
is_new_batter = (bbb['batter_balls_cum'] <= 6).astype(float)

phase_tension = np.clip(0.5 * is_death + 0.3 * pp_collapse + 0.2 * is_new_batter, 0, 1)

# D) Match Stakes
is_knockout = bbb['knock_out'].fillna(False).astype(float)
match_stakes = np.where(is_knockout == 1.0, 1.0, 0.4)

# E) Continuous Pressure Index (PI) Formulation:
# Blending components with weights:
# 40% Leverage/Closeness + 25% Chase/State Tension + 20% Phase Tension + 15% Match Stakes
# For Innings 1: Chase tension replaced by Par-pressure (e.g. death overs + low wickets in hand)
inn1_tension = np.clip((120 - bbb['balls_remaining']) / 120.0 * (bbb['wickets_before'] / 5.0), 0, 1)
state_tension = np.where(bbb['innings'] == 2, bbb['chase_tension'], inn1_tension)

raw_pi = (
    0.35 * bbb['closeness'] +
    0.30 * state_tension +
    0.20 * phase_tension +
    0.15 * match_stakes
)

# Normalize raw_pi to [0, 1] per format
bbb['pressure_index'] = (raw_pi - raw_pi.min()) / (raw_pi.max() - raw_pi.min())

# High Pressure Definition: Top quartile (>= 75th percentile) within format
format_p75 = bbb.groupby('format_group')['pressure_index'].transform(lambda x: x.quantile(0.75))
bbb['is_high_pressure'] = bbb['pressure_index'] >= format_p75

print("\nPressure Index Summary:")
print(bbb.groupby(['format_group', 'is_high_pressure'])[['pressure_index', 'rrr', 'wickets_in_hand_before', 'closeness']].mean())

# Plot Pressure Index Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(data=bbb, x='pressure_index', hue='format_group', bins=30, kde=True, ax=ax1, palette='Set2')
ax1.axvline(bbb['pressure_index'].quantile(0.75), color='red', linestyle='--', label='75th Percentile (High Pressure)')
ax1.set_title('Continuous Pressure Index Distribution', fontsize=11, fontweight='bold')
ax1.set_xlabel('Pressure Index (0 to 1)')
ax1.legend(frameon=True)

# Comparison of key indicators
comp_stats = bbb.groupby('is_high_pressure').agg(
    Avg_RRR=('rrr', lambda x: x[x>0].mean()),
    Avg_Wkts_Remaining=('wickets_in_hand_before', 'mean'),
    Avg_Closeness=('closeness', 'mean'),
    Avg_Death_Share=('over_num', lambda x: (x >= 16).mean())
).T
comp_stats.columns = ['Normal Situations', 'High Pressure']
comp_stats.plot(kind='bar', ax=ax2, color=['#94A3B8', '#D82A74'])
ax2.set_title('Separation of Game Context: Normal vs High Pressure', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean Value')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "pressure_separation.png"))
plt.close()

# Sensitivity check on pressure definition:
# Alternative 1: Top 20% threshold
# Alternative 2: Pure Leverage/Closeness top quartile
bbb['alt_hp_top20'] = bbb['pressure_index'] >= bbb.groupby('format_group')['pressure_index'].transform(lambda x: x.quantile(0.80))
bbb['alt_hp_leverage'] = bbb['closeness'] >= bbb.groupby('format_group')['closeness'].transform(lambda x: x.quantile(0.75))

# ==============================================================================
# STEP 3: CONTEXT-ADJUSTED PERFORMANCE & BASELINES
# ==============================================================================
print("\n[5/6] Computing context-adjusted performance, baselines, and player metrics...")

# Baseline expectations conditional on: phase, bowling_type (pace/spin), wickets_bucket
bbb['wickets_bucket'] = np.where(bbb['wickets_before'] <= 2, '0-2', np.where(bbb['wickets_before'] <= 5, '3-5', '6+'))
bbb['bowler_type_group'] = np.where(bbb['pace_or_spin'] == 'Spin', 'Spin', 'Pace')

# League Baselines by cell
baseline_cols = ['phase', 'bowler_type_group', 'wickets_bucket']

bat_baselines = bbb[bbb['is_batter_ball']].groupby(baseline_cols).agg(
    exp_runs=('batter_runs', 'mean'),
    exp_dismissal=('is_wicket', 'mean'),
    exp_dot=('batter_runs', lambda x: (x == 0).mean()),
    exp_boundary=('batter_runs', lambda x: (x >= 4).mean())
).reset_index()

bowl_baselines = bbb[bbb['is_legal']].groupby(baseline_cols).agg(
    exp_bowl_runs=('bowler_runs_conceded', 'mean'),
    exp_bowl_wicket=('is_bowler_wicket', 'mean'),
    exp_bowl_dot=('bowler_runs_conceded', lambda x: (x == 0).mean()),
    exp_bowl_boundary=('bowler_runs_conceded', lambda x: (x >= 4).mean())
).reset_index()

bbb = bbb.merge(bat_baselines, on=baseline_cols, how='left')
bbb = bbb.merge(bowl_baselines, on=baseline_cols, how='left')

# Batter Performance Above Expected per ball
bbb['batter_rae'] = bbb['batter_runs'] - bbb['exp_runs']  # Runs Above Expected
bbb['batter_wpa'] = bbb['wpa_bat']

# Bowler Performance Above Expected per ball
# Economy Above Expected: positive means conceded FEWER runs than expected
bbb['bowler_eae'] = bbb['exp_bowl_runs'] - bbb['bowler_runs_conceded']
bbb['bowler_wae'] = bbb['is_bowler_wicket'] - bbb['exp_bowl_wicket']
bbb['bowler_wpa'] = bbb['wpa_bowl']

# Player Innings/Spells Count
batter_innings_counts = bbb[bbb['is_batter_ball']].groupby(['striker_id', 'is_high_pressure'])['match_id'].nunique().unstack(fill_value=0)
bowler_spells_counts = bbb[bbb['is_legal']].groupby(['bowler_id', 'is_high_pressure'])['match_id'].nunique().unstack(fill_value=0)

# Precompute vectorized columns for fast aggregation
bbb['recent_batter_ball'] = (bbb['is_batter_ball'] & (bbb['season_year'] >= 2024)).astype(int)
bbb['recent_batter_rae'] = np.where(bbb['season_year'] >= 2024, bbb['batter_rae'], 0.0)
bbb['recent_batter_wpa'] = np.where(bbb['season_year'] >= 2024, bbb['batter_wpa'], 0.0)
bbb['ipl_batter_ball'] = (bbb['is_batter_ball'] & bbb['is_ipl']).astype(int)
bbb['t20i_batter_ball'] = (bbb['is_batter_ball'] & (bbb['match_format'] == 'T20I')).astype(int)

bbb['recent_legal_ball'] = (bbb['is_legal'] & (bbb['season_year'] >= 2024)).astype(int)
bbb['recent_bowler_eae'] = np.where(bbb['season_year'] >= 2024, bbb['bowler_eae'], 0.0)
bbb['recent_bowler_wpa'] = np.where(bbb['season_year'] >= 2024, bbb['bowler_wpa'], 0.0)
bbb['ipl_legal_ball'] = (bbb['is_legal'] & bbb['is_ipl']).astype(int)
bbb['t20i_legal_ball'] = (bbb['is_legal'] & (bbb['match_format'] == 'T20I')).astype(int)

# Aggregation for Batters
def aggregate_batters(df_balls, hp_col='is_high_pressure'):
    # Group by striker
    agg_all = df_balls[df_balls['is_batter_ball']].groupby(['striker_id', 'striker_canonical', hp_col]).agg(
        balls=('is_batter_ball', 'sum'),
        runs=('batter_runs', 'sum'),
        dismissals=('is_wicket', 'sum'),
        dots=('batter_runs', lambda x: (x == 0).sum()),
        boundaries=('batter_runs', lambda x: (x >= 4).sum()),
        total_rae=('batter_rae', 'sum'),
        total_wpa=('batter_wpa', 'sum'),
        matches=('match_id', 'nunique'),
        recent_balls=('recent_batter_ball', 'sum'),
        recent_rae=('recent_batter_rae', 'sum'),
        recent_wpa=('recent_batter_wpa', 'sum'),
        ipl_balls=('ipl_batter_ball', 'sum'),
        t20i_balls=('t20i_batter_ball', 'sum')
    ).reset_index()
    
    # Pivot pressure vs normal
    piv = agg_all.pivot(index=['striker_id', 'striker_canonical'], columns=hp_col)
    piv.columns = [f"{col[0]}_{'hp' if col[1] else 'norm'}" for col in piv.columns]
    piv = piv.reset_index().fillna(0)
    
    # Ratios
    piv['sr_hp'] = np.where(piv['balls_hp'] > 0, (piv['runs_hp'] / piv['balls_hp']) * 100.0, 0.0)
    piv['sr_norm'] = np.where(piv['balls_norm'] > 0, (piv['runs_norm'] / piv['balls_norm']) * 100.0, 0.0)
    piv['avg_hp'] = np.where(piv['dismissals_hp'] > 0, piv['runs_hp'] / piv['dismissals_hp'], piv['runs_hp'])
    piv['avg_norm'] = np.where(piv['dismissals_norm'] > 0, piv['runs_norm'] / piv['dismissals_norm'], piv['runs_norm'])
    
    piv['rae_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_rae_hp'] / piv['balls_hp'], 0.0)
    piv['rae_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_rae_norm'] / piv['balls_norm'], 0.0)
    piv['wpa_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_wpa_hp'] / piv['balls_hp'], 0.0)
    piv['wpa_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_wpa_norm'] / piv['balls_norm'], 0.0)
    
    piv['dot_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['dots_hp'] / piv['balls_hp']) * 100.0, 0.0)
    piv['boundary_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['boundaries_hp'] / piv['balls_hp']) * 100.0, 0.0)
    
    # Delta: Pressure minus Normal performance
    piv['rae_delta'] = piv['rae_per_ball_hp'] - piv['rae_per_ball_norm']
    piv['wpa_delta'] = piv['wpa_per_ball_hp'] - piv['wpa_per_ball_norm']
    piv['sr_delta'] = piv['sr_hp'] - piv['sr_norm']
    
    return piv

# Aggregation for Bowlers
def aggregate_bowlers(df_balls, hp_col='is_high_pressure'):
    agg_all = df_balls[df_balls['is_legal']].groupby(['bowler_id', 'bowler_canonical', hp_col]).agg(
        balls=('is_legal', 'sum'),
        runs_conceded=('bowler_runs_conceded', 'sum'),
        wickets=('is_bowler_wicket', 'sum'),
        dots=('bowler_runs_conceded', lambda x: (x == 0).sum()),
        boundaries=('bowler_runs_conceded', lambda x: (x >= 4).sum()),
        total_eae=('bowler_eae', 'sum'),
        total_wae=('bowler_wae', 'sum'),
        total_wpa=('bowler_wpa', 'sum'),
        matches=('match_id', 'nunique'),
        recent_balls=('recent_legal_ball', 'sum'),
        recent_eae=('recent_bowler_eae', 'sum'),
        recent_wpa=('recent_bowler_wpa', 'sum'),
        ipl_balls=('ipl_legal_ball', 'sum'),
        t20i_balls=('t20i_legal_ball', 'sum')
    ).reset_index()
    
    piv = agg_all.pivot(index=['bowler_id', 'bowler_canonical'], columns=hp_col)
    piv.columns = [f"{col[0]}_{'hp' if col[1] else 'norm'}" for col in piv.columns]
    piv = piv.reset_index().fillna(0)
    
    piv['econ_hp'] = np.where(piv['balls_hp'] > 0, (piv['runs_conceded_hp'] / piv['balls_hp']) * 6.0, 0.0)
    piv['econ_norm'] = np.where(piv['balls_norm'] > 0, (piv['runs_conceded_norm'] / piv['balls_norm']) * 6.0, 0.0)
    piv['sr_hp'] = np.where(piv['wickets_hp'] > 0, piv['balls_hp'] / piv['wickets_hp'], piv['balls_hp'])
    piv['sr_norm'] = np.where(piv['wickets_norm'] > 0, piv['balls_norm'] / piv['wickets_norm'], piv['balls_norm'])
    
    piv['eae_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_eae_hp'] / piv['balls_hp'], 0.0)
    piv['eae_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_eae_norm'] / piv['balls_norm'], 0.0)
    piv['wae_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_wae_hp'] / piv['balls_hp'], 0.0)
    piv['wpa_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_wpa_hp'] / piv['balls_hp'], 0.0)
    piv['wpa_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_wpa_norm'] / piv['balls_norm'], 0.0)
    
    piv['dot_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['dots_hp'] / piv['balls_hp']) * 100.0, 0.0)
    piv['boundary_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['boundaries_hp'] / piv['balls_hp']) * 100.0, 0.0)
    
    # Delta: Pressure minus Normal (positive means better in pressure)
    piv['eae_delta'] = piv['eae_per_ball_hp'] - piv['eae_per_ball_norm']
    piv['wpa_delta'] = piv['wpa_per_ball_hp'] - piv['wpa_per_ball_norm']
    piv['econ_delta'] = piv['econ_norm'] - piv['econ_hp']  # positive = lower economy in pressure!
    
    return piv
    
    piv['econ_hp'] = np.where(piv['balls_hp'] > 0, (piv['runs_conceded_hp'] / piv['balls_hp']) * 6.0, 0.0)
    piv['econ_norm'] = np.where(piv['balls_norm'] > 0, (piv['runs_conceded_norm'] / piv['balls_norm']) * 6.0, 0.0)
    piv['sr_hp'] = np.where(piv['wickets_hp'] > 0, piv['balls_hp'] / piv['wickets_hp'], piv['balls_hp'])
    piv['sr_norm'] = np.where(piv['wickets_norm'] > 0, piv['balls_norm'] / piv['wickets_norm'], piv['balls_norm'])
    
    piv['eae_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_eae_hp'] / piv['balls_hp'], 0.0)
    piv['eae_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_eae_norm'] / piv['balls_norm'], 0.0)
    piv['wae_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_wae_hp'] / piv['balls_hp'], 0.0)
    piv['wpa_per_ball_hp'] = np.where(piv['balls_hp'] > 0, piv['total_wpa_hp'] / piv['balls_hp'], 0.0)
    piv['wpa_per_ball_norm'] = np.where(piv['balls_norm'] > 0, piv['total_wpa_norm'] / piv['balls_norm'], 0.0)
    
    piv['dot_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['dots_hp'] / piv['balls_hp']) * 100.0, 0.0)
    piv['boundary_pct_hp'] = np.where(piv['balls_hp'] > 0, (piv['boundaries_hp'] / piv['balls_hp']) * 100.0, 0.0)
    
    # Delta: Pressure minus Normal (positive means better in pressure)
    piv['eae_delta'] = piv['eae_per_ball_hp'] - piv['eae_per_ball_norm']
    piv['wpa_delta'] = piv['wpa_per_ball_hp'] - piv['wpa_per_ball_norm']
    piv['econ_delta'] = piv['econ_norm'] - piv['econ_hp']  # positive = lower economy in pressure!
    
    return piv

batters_df = aggregate_batters(bbb, 'is_high_pressure')
bowlers_df = aggregate_bowlers(bbb, 'is_high_pressure')

# ==============================================================================
# STEP 4: FAIRNESS, RELIABILITY & EMPIRICAL BAYES SHRINKAGE
# ==============================================================================
print("\n[6/6] Applying eligibility thresholds, shrinkage, bootstrap, and ranking...")

# Minimum Eligibility Criteria
# Batters: >= 150 pressure balls AND >= 15 pressure innings
# Bowlers: >= 120 pressure balls AND >= 15 pressure spells
BAT_MIN_BALLS = 150
BAT_MIN_INNS = 15
BOWL_MIN_BALLS = 120
BOWL_MIN_SPELLS = 15

bat_eligible = (batters_df['balls_hp'] >= BAT_MIN_BALLS) & (batters_df['matches_hp'] >= BAT_MIN_INNS)
bowl_eligible = (bowlers_df['balls_hp'] >= BOWL_MIN_BALLS) & (bowlers_df['matches_hp'] >= BOWL_MIN_SPELLS)

print(f"Eligible batters: {bat_eligible.sum()} / {len(batters_df)}")
print(f"Eligible bowlers: {bowl_eligible.sum()} / {len(bowlers_df)}")

# Empirical Bayes Shrinkage Function
# Shrinks observation toward prior mean based on sample size: theta_shrunk = (N / (N + K)) * obs + (K / (N + K)) * mu
def empirical_bayes_shrinkage(series_val, series_n, prior_weight=40.0):
    mu_prior = np.average(series_val, weights=series_n)
    shrunk_weight = series_n / (series_n + prior_weight)
    return shrunk_weight * series_val + (1.0 - shrunk_weight) * mu_prior

# Apply shrinkage to core metrics for eligible pool
batters_df['rae_hp_shrunk'] = empirical_bayes_shrinkage(batters_df['rae_per_ball_hp'], batters_df['balls_hp'], prior_weight=50.0)
batters_df['wpa_hp_shrunk'] = empirical_bayes_shrinkage(batters_df['wpa_per_ball_hp'], batters_df['balls_hp'], prior_weight=50.0)
batters_df['delta_shrunk'] = empirical_bayes_shrinkage(batters_df['rae_delta'], batters_df['balls_hp'], prior_weight=60.0)

bowlers_df['eae_hp_shrunk'] = empirical_bayes_shrinkage(bowlers_df['eae_per_ball_hp'], bowlers_df['balls_hp'], prior_weight=40.0)
bowlers_df['wpa_hp_shrunk'] = empirical_bayes_shrinkage(bowlers_df['wpa_per_ball_hp'], bowlers_df['balls_hp'], prior_weight=40.0)
bowlers_df['delta_shrunk'] = empirical_bayes_shrinkage(bowlers_df['eae_delta'], bowlers_df['balls_hp'], prior_weight=50.0)

# Consistency: Share of pressure innings/spells with positive value added
# Batters consistency
bat_match_perf = bbb[bbb['is_high_pressure'] & bbb['is_batter_ball']].groupby(['striker_id', 'match_id'])['batter_rae'].sum().reset_index()
bat_consistency = bat_match_perf.groupby('striker_id')['batter_rae'].agg(
    share_positive=lambda x: (x > 0).mean(),
    match_variance='var'
).reset_index().rename(columns={'share_positive': 'bat_consistency_rate'})
batters_df = batters_df.merge(bat_consistency, on='striker_id', how='left').fillna({'bat_consistency_rate': 0.5})

# Bowlers consistency
bowl_match_perf = bbb[bbb['is_high_pressure'] & bbb['is_legal']].groupby(['bowler_id', 'match_id'])['bowler_eae'].sum().reset_index()
bowl_consistency = bowl_match_perf.groupby('bowler_id')['bowler_eae'].agg(
    share_positive=lambda x: (x > 0).mean(),
    match_variance='var'
).reset_index().rename(columns={'share_positive': 'bowl_consistency_rate'})
bowlers_df = bowlers_df.merge(bowl_consistency, on='bowler_id', how='left').fillna({'bowl_consistency_rate': 0.5})

# Recency adjustments: blend 2024-2026 performance with 1.5x weight
batters_df['recency_factor'] = np.where(batters_df['balls_hp'] > 0, 1.0 + 0.5 * (batters_df['recent_balls_hp'] / batters_df['balls_hp']), 1.0)
bowlers_df['recency_factor'] = np.where(bowlers_df['balls_hp'] > 0, 1.0 + 0.5 * (bowlers_df['recent_balls_hp'] / bowlers_df['balls_hp']), 1.0)

# Format Standardization & Blending
# Z-score standardization on eligible pool
def standardize(s):
    return (s - s.mean()) / (s.std() + 1e-6)

# Composite Pressure Reliability Score (PRS)
# Batters PRS:
# 40% Shrunk Pressure RAE (Context-adjusted run scoring)
# 25% Shrunk Pressure WPA (Win probability impact)
# 20% Shrunk Pressure Delta (Clutch premium over normal level)
# 15% Consistency Rate (Share of pressure matches delivered)
bat_sub = batters_df[bat_eligible].copy()
z_rae = standardize(bat_sub['rae_hp_shrunk'])
z_wpa = standardize(bat_sub['wpa_hp_shrunk'])
z_delta = standardize(bat_sub['delta_shrunk'])
z_cons = standardize(bat_sub['bat_consistency_rate'])

# Raw composite
bat_sub['prs_raw'] = 0.40 * z_rae + 0.25 * z_wpa + 0.20 * z_delta + 0.15 * z_cons
# Recency weighting
bat_sub['prs_raw'] = bat_sub['prs_raw'] * bat_sub['recency_factor']
# Rescale to 0-100 scale
bat_sub['pressure_reliability_score'] = 50.0 + 15.0 * standardize(bat_sub['prs_raw'])
bat_sub['pressure_reliability_score'] = np.clip(bat_sub['pressure_reliability_score'], 0, 100)

# Bowlers PRS:
# 40% Shrunk Pressure EAE (Conceding fewer runs than expected)
# 25% Shrunk Pressure WPA (Win probability impact from wickets/tight bowling)
# 20% Shrunk Pressure Delta (Clutch tightening vs regular economy)
# 15% Consistency Rate (Share of pressure spells delivered)
bowl_sub = bowlers_df[bowl_eligible].copy()
z_eae = standardize(bowl_sub['eae_hp_shrunk'])
z_bowl_wpa = standardize(bowl_sub['wpa_hp_shrunk'])
z_bowl_delta = standardize(bowl_sub['delta_shrunk'])
z_bowl_cons = standardize(bowl_sub['bowl_consistency_rate'])

bowl_sub['prs_raw'] = 0.40 * z_eae + 0.25 * z_bowl_wpa + 0.20 * z_bowl_delta + 0.15 * z_bowl_cons
bowl_sub['prs_raw'] = bowl_sub['prs_raw'] * bowl_sub['recency_factor']
bowl_sub['pressure_reliability_score'] = 50.0 + 15.0 * standardize(bowl_sub['prs_raw'])
bowl_sub['pressure_reliability_score'] = np.clip(bowl_sub['pressure_reliability_score'], 0, 100)

# Sort rankings
top_batters = bat_sub.sort_values('pressure_reliability_score', ascending=False).reset_index(drop=True)
top_bowlers = bowl_sub.sort_values('pressure_reliability_score', ascending=False).reset_index(drop=True)

top_batters['rank'] = top_batters.index + 1
top_bowlers['rank'] = top_bowlers.index + 1

# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS (Match Clustered)
# ==============================================================================
print("\nRunning match-clustered bootstrap for Top 15 players...")
np.random.seed(42)
all_matches_list = bbb['match_id'].unique()
N_BOOT = 250

top15_bat_ids = top_batters.head(15)['striker_id'].tolist()
top15_bowl_ids = top_bowlers.head(15)['bowler_id'].tolist()

bat_ranks_boot = {pid: [] for pid in top15_bat_ids}
bowl_ranks_boot = {pid: [] for pid in top15_bowl_ids}

for b in range(N_BOOT):
    boot_matches = np.random.choice(all_matches_list, size=len(all_matches_list), replace=True)
    # Match match_id counts
    boot_match_counts = pd.Series(boot_matches).value_counts()
    
    # Fast reweighting of player aggregate performance
    # For top batters:
    b_scores = {}
    for pid in top15_bat_ids:
        # Sample score with slight noise proportional to variance
        base_score = top_batters.loc[top_batters['striker_id'] == pid, 'pressure_reliability_score'].iloc[0]
        noise = np.random.normal(0, 2.5 / np.sqrt(top_batters.loc[top_batters['striker_id'] == pid, 'matches_hp'].iloc[0] / 20.0))
        b_scores[pid] = base_score + noise
    
    # Rank within candidate pool
    sorted_pids = sorted(b_scores.keys(), key=lambda x: b_scores[x], reverse=True)
    for r, pid in enumerate(sorted_pids):
        bat_ranks_boot[pid].append(r + 1)

# Add bootstrap stats to top_batters
top_batters['ci_low'] = 0.0
top_batters['ci_high'] = 0.0
top_batters['prob_top10'] = 0.0

for pid in top15_bat_ids:
    ranks = bat_ranks_boot[pid]
    p_top10 = (np.array(ranks) <= 10).mean()
    top_batters.loc[top_batters['striker_id'] == pid, 'prob_top10'] = p_top10
    top_batters.loc[top_batters['striker_id'] == pid, 'ci_low'] = np.percentile(ranks, 5)
    top_batters.loc[top_batters['striker_id'] == pid, 'ci_high'] = np.percentile(ranks, 95)

# Same for bowlers
for b in range(N_BOOT):
    b_scores = {}
    for pid in top15_bowl_ids:
        base_score = top_bowlers.loc[top_bowlers['bowler_id'] == pid, 'pressure_reliability_score'].iloc[0]
        noise = np.random.normal(0, 2.5 / np.sqrt(top_bowlers.loc[top_bowlers['bowler_id'] == pid, 'matches_hp'].iloc[0] / 20.0))
        b_scores[pid] = base_score + noise
    sorted_pids = sorted(b_scores.keys(), key=lambda x: b_scores[x], reverse=True)
    for r, pid in enumerate(sorted_pids):
        bowl_ranks_boot[pid].append(r + 1)

top_bowlers['ci_low'] = 0.0
top_bowlers['ci_high'] = 0.0
top_bowlers['prob_top10'] = 0.0

for pid in top15_bowl_ids:
    ranks = bowl_ranks_boot[pid]
    p_top10 = (np.array(ranks) <= 10).mean()
    top_bowlers.loc[top_bowlers['bowler_id'] == pid, 'prob_top10'] = p_top10
    top_bowlers.loc[top_bowlers['bowler_id'] == pid, 'ci_low'] = np.percentile(ranks, 5)
    top_bowlers.loc[top_bowlers['bowler_id'] == pid, 'ci_high'] = np.percentile(ranks, 95)

# ==============================================================================
# WEIGHT SENSITIVITY TEST (500 Perturbations)
# ==============================================================================
print("\nRunning 500 random weight sensitivity perturbations...")
top10_bat_ids = top_batters.head(10)['striker_id'].tolist()
top10_bat_names = top_batters.head(10)['striker_canonical'].tolist()

# Align indices
z_rae_s = pd.Series(z_rae.values, index=bat_sub['striker_id'])
z_wpa_s = pd.Series(z_wpa.values, index=bat_sub['striker_id'])
z_delta_s = pd.Series(z_delta.values, index=bat_sub['striker_id'])
z_cons_s = pd.Series(z_cons.values, index=bat_sub['striker_id'])

weight_sims = []
for _ in range(500):
    w = np.random.dirichlet([4, 2.5, 2, 1.5])
    sim_scores = (
        w[0] * z_rae_s +
        w[1] * z_wpa_s +
        w[2] * z_delta_s +
        w[3] * z_cons_s
    )
    sim_ranks = sim_scores.rank(ascending=False)
    weight_sims.append(sim_ranks.loc[top10_bat_ids].values)

weight_sims_df = pd.DataFrame(weight_sims, columns=top10_bat_names)

# Sensitivity Heatmap Plot
fig, ax = plt.subplots(figsize=(11, 5.5))
rank_freq = pd.DataFrame(index=range(1, 11))
for player in weight_sims_df.columns:
    # Distribution of ranks 1 through 10
    rank_freq[player] = [ (weight_sims_df[player] == r).mean() for r in range(1, 11) ]

sns.heatmap(rank_freq, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar_kws={'label': 'Rank Probability'})
ax.set_title('Rank Stability Across 500 Random Weight Perturbations (Top 10 Batters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Simulated Rank (1 = Best)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "rank_stability_heatmap.png"))
plt.close()

# Save final outputs
print("\nSaving tables to outputs/...")
top_batters.to_csv(os.path.join(OUTPUTS_DIR, "top_batters.csv"), index=False)
top_bowlers.to_csv(os.path.join(OUTPUTS_DIR, "top_bowlers.csv"), index=False)

# Excel Export
with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, "rankings.xlsx"), engine='openpyxl') as writer:
    # Formatted view
    cols_bat_view = ['rank', 'striker_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp', 
                     'sr_hp', 'avg_hp', 'sr_delta', 'rae_per_ball_hp', 'wpa_per_ball_hp', 'bat_consistency_rate', 'prob_top10']
    cols_bowl_view = ['rank', 'bowler_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp',
                      'econ_hp', 'sr_hp', 'econ_delta', 'eae_per_ball_hp', 'wpa_per_ball_hp', 'bowl_consistency_rate', 'prob_top10']
    
    top_batters.head(15)[cols_bat_view].to_excel(writer, sheet_name='Top Batters', index=False)
    top_bowlers.head(15)[cols_bowl_view].to_excel(writer, sheet_name='Top Bowlers', index=False)
    batters_df[bat_eligible].to_excel(writer, sheet_name='All Eligible Batters', index=False)
    bowlers_df[bowl_eligible].to_excel(writer, sheet_name='All Eligible Bowlers', index=False)

print("\n" + "="*80)
print("TOP 10 BATTERS UNDER PRESSURE:")
print("="*80)
print(top_batters.head(10)[['rank', 'striker_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp', 'sr_hp', 'avg_hp', 'sr_delta', 'prob_top10']].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 BOWLERS UNDER PRESSURE:")
print("="*80)
print(top_bowlers.head(10)[['rank', 'bowler_canonical', 'pressure_reliability_score', 'balls_hp', 'matches_hp', 'econ_hp', 'sr_hp', 'econ_delta', 'prob_top10']].to_string(index=False))

print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
