from lib.nba_dataset import load_nba, update_combined, load_rookies, filter_dataset
from lib.mlflow import log_to_mlflow, get_experiment_and_model_name
from lib.check_score import calculate_score, load_json_file
from lib.metric import get_weighted_team_predictions, custom_score_func, get_corr_charts

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import KNNImputer

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from functools import partial


START = 2018
LEARN = 6
TEST  = '2025'
GROUND_TRUTH_NBA = "output/ground_truth.json"
PREDICTED_NBA = "output/predicted.json"
RANDOM_STATE = 42
seasons = list(map(str, range(START, START+LEARN, 1)))

nba = [Bunch() for _ in seasons]

top_indices = [[],[],[],[]]
results_dict = {}
true_results_dict = {}

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

nba_predict = load_nba(TEST)
for season_id, season in enumerate(seasons):
    nba[season_id] = load_nba(season)
    nba_combined = update_combined(nba_combined, nba[season_id])

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 95, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [50, 100, 200, 300, 500]
# }
# param_grid = {
#     'max_depth': [2, 5, 10, 50],
#     'learning_rate': [0.01, 0.1],
#     'subsample': [0.5, 1.0],
#     'min_samples_split': [2, 5, 10],
#     'n_estimators': [50, 100, 200, 300, 500]
# }
param_grid = {
    'bootstrap': [True],
    'max_depth': Integer(20, 500),
    'max_features': Integer(2, 10),
    'min_samples_leaf': Integer(2, 10),
    'min_samples_split': Integer(5, 20),
    'n_estimators': Integer(50, 500)
}
# param_grid = {
#     'n_estimators': Integer(20, 400),
#     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'max_depth': Integer(2, 10),
#     'subsample': Real(0.3, 1.0),
#     'min_samples_split': Integer(2, 10),
#     'max_features': Integer(10,40),
# }

skf = StratifiedKFold(n_splits=3, shuffle=True)
rfc = RandomForestClassifier(random_state=RANDOM_STATE)
gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)

imputer = KNNImputer(n_neighbors=5)
nba_combined = filter_dataset(
    nba_combined,
    rookies=False,
    fill_na=lambda col: pd.Series(imputer.fit_transform(col.values.reshape(-1, 1)).flatten(), index=col.index) if col.isnull().any() else col,
    debug=True,
    drop_low_chances=True
)
nba_predict = filter_dataset(
    nba_predict,
    rookies=False,
    fill_na=lambda col: pd.Series(imputer.transform(col.values.reshape(-1, 1)).flatten(), index=col.index) if col.isnull().any() else col,
    debug=True,
    drop_low_chances=True
)

charts_dict = get_corr_charts(nba_combined)

custom_scorer = partial(custom_score_func, record=nba_combined.record)
clf = BayesSearchCV(
    rfc,
    param_grid,
    scoring={
        'custom': custom_scorer,
        'recall_macro': make_scorer(recall_score, average='macro')
    },
    refit='custom',
    cv=skf,
    n_jobs=-1
)

weights = compute_sample_weight(class_weight='balanced', y=nba_combined.target)
clf.fit(nba_combined.data, nba_combined.target.values.flatten().astype(int), sample_weight=weights)
# clf.fit(nba_combined.data, nba_combined.target.values.flatten().astype(int))

probs = clf.predict_proba(nba_predict.data)
results_dict, true_results_dict, players_team, true_team = get_weighted_team_predictions(
    probs, nba_predict.target, nba_predict.record
)
score = calculate_score(results_dict, true_results_dict)
max_score = calculate_score(true_results_dict, true_results_dict)
print(f'Your score: {score} / {max_score}!')

cm_plot, ax = plt.subplots()
cm = confusion_matrix(true_team, players_team, labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot(ax=ax)
charts_dict["confusion_matrix"] = cm_plot

# fig, ax = plt.subplots()
# ax.hist(weights, bins=20)
# ax.set_title("Sample Weights Distribution")
# charts_dict["sample_weights_distribution"] = fig

with open(PREDICTED_NBA, "w") as f:
    json.dump(results_dict, f, indent=2)
with open(GROUND_TRUTH_NBA, "w") as f:
    json.dump(true_results_dict, f, indent=2)



experiment_name, model_name = get_experiment_and_model_name(clf, max_score)

log_to_mlflow(
    clf,
    experiment_name,
    model_name,
    params=clf.get_params(),
    metrics={
        "max_score": max_score,
        "score": score,
        "score_perc": score / max_score * 100
    },
    input_data=nba_predict.data,
    charts=charts_dict,
    artifacts={
        "predicted.json": PREDICTED_NBA,
        "ground_truth.json": GROUND_TRUTH_NBA
    }
)


a =  5
