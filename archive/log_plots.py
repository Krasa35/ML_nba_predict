from lib.nba_dataset import load_nba, update_combined, load_rookies
from lib.mlflow import log_to_mlflow
from lib.check_score import calculate_score, load_json_file
from lib.metric import get_weighted_team_predictions, custom_score_func, get_corr_based_plot, get_corr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


START = 2000
LEARN = 25
TEST  = '2025'
GROUND_TRUTH_NBA = "output/ground_truth.json"
PREDICTED_NBA = "output/predicted.json"
seasons = list(map(str, range(START, START+LEARN, 1)))
cols_to_drop_NBA = [
    'MP_totadv', 'Rk_advanced',
    'FG_totadv', 'FGA_totadv', 'PTS_totadv',
    '2P_totadv', '2PA_totadv',
    'FT_totadv', 'FTA_totadv',
    'ORB_totadv', 'DRB_totadv', 'TRB_totadv',
    'TOV_totadv', 'AST_totadv', 'BLK_totadv',
    'FG_per_game', 'FGA_per_game',
    '2P_per_game', '2PA_per_game',
    '3P_totadv', '3PA_totadv', '3P_per_game', '3PA_per_game',
    'FT_per_game', 'FTA_per_game',
    'Rk_totadvpg', 'OBPM', 'WS',
    '3PAr','TRB%', 'W_team', 'L_team',
    'FG% by Distance_3P', 'eFG%',               
    'DRB_per_game', 'Rk_shooting',
    '% of FGA by Distance_2P', 'W/L%_team', "2P%"
]
cols_to_drop_rookies = [
    'MP_totadv', 'FG_totadv',       
    'FGA_totadv', '2P_totadv',
    '2PA_totadv', 'PTS_totadv',    
    '3PA_totadv', 'FT_totadv',
    'FTA_totadv', 'TRB_totadv',
    'DRB_totadv', 'ORB_totadv',
    'TOV_totadv', 'PF_totadv',
    'Rk_advanced', 'Rk_totadvpg',
    'eFG%', '% of FGA by Distance_2P',
    'WS/48', 'OBPM',
    '3PA_per_game','2PA_per_game',
    'PTS_per_game', 'Rk_shooting',
    'FG% by Distance_3P', 'FGA_per_game',
    '2P_per_game', 'FTA_per_game',
    'DRB_per_game', '2P%'
]

nba = [Bunch() for _ in seasons]

nba_combined = Bunch(
        data=pd.DataFrame(),
        target=pd.DataFrame(),
        record=pd.DataFrame(),
        frame=pd.DataFrame(),
        target_names=str,   
        DESCR=str,
        feature_names=np.ndarray,
        data_module="sklearn.datasets.data",
    )

for season_id, season in enumerate(seasons):
    nba[season_id] = load_rookies(season)
    nba_combined = update_combined(nba_combined, nba[season_id])

# nba_combined.data = nba_combined.data.drop(columns=cols_to_drop, errors='ignore')
nba_combined.data = nba_combined.data.drop(columns=cols_to_drop_rookies, errors='ignore')
corr = get_corr_matrix(nba_combined, only_super_team=True)
corr_plot = get_corr_based_plot(corr, plot_type=10)
anti_corr_plot = get_corr_based_plot(corr, plot_type=20)
f1_vs_target_plot = get_corr_based_plot(corr, plot_type=30, df = nba_combined)
f2_vs_target_plot = get_corr_based_plot(corr, plot_type=31, df = nba_combined)
f3_vs_target_plot = get_corr_based_plot(corr, plot_type=32, df = nba_combined)
a=5

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

skf = StratifiedKFold(n_splits=3)
clf = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    scoring={
        'custom': custom_score_func,
        'recall_macro': make_scorer(recall_score, average='macro')
    },
    refit='recall_macro',
    cv=skf,
    n_jobs=-1
)

experiment_name = 'data_correlations'
model_name = 'CORRELATION_allRookieTeam_Rookies'

corr = get_corr_matrix(nba_combined, only_super_team=True, return_whole=True)
threshold = 0.9
corr_pairs = corr.abs().unstack()
# Remove self-correlations
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
# Keep only one of each pair
corr_pairs = corr_pairs.drop_duplicates()
high_corr_pairs = corr_pairs[(corr_pairs > threshold) | (corr_pairs < -threshold)]

with open("output/high_corr_pairs.txt", "w") as f:
    if not high_corr_pairs.empty:
        f.write(f"Feature pairs with correlation magnitude > {threshold}\n")
        f.write(f"Found {len(high_corr_pairs)} highly correlated pairs:\n")
        for (feat1, feat2), corr_val in high_corr_pairs.items():
            f.write(f"{feat1} & {feat2}: {corr_val:.3f}\n")
    else:
        f.write(f"No feature pairs with correlation magnitude > {threshold}.\n")

log_to_mlflow(
    clf,
    experiment_name,
    model_name,
    params=clf.get_params(),
    charts={
        "feature 1 vs target corr": f1_vs_target_plot,
        "feature 2 vs target corr": f2_vs_target_plot,
        "feature 3 vs target corr": f3_vs_target_plot,
        "correlation plot": corr_plot,
        "anti correlation plot": anti_corr_plot,
        },
    artifacts={
        # "gs_comparison.txt": "output/gs_comparison.txt",
        "high_corr_pairs.txt": "output/high_corr_pairs.txt"
    }
)
