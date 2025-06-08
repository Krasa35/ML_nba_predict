from lib.nba_dataset import load_nba, update_combined

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from lib.mlflow import log_to_mlflow
from lib.check_score import calculate_score, load_json_file

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import tempfile
import os


START = 2020
LEARN = 3
TEST  = '2023-24'
GROUND_TRUTH_NBA = "output/ground_truth.json"
PREDICTED_NBA = "output/predicted.json"
seasons = [f"{START + n}-{str(START + n + 1 - int(START / 100) * 100).zfill(2)}" for n in range(LEARN)]

nba = [Bunch() for _ in range(LEARN)]

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
# Update the loop
for season_id, season in enumerate(seasons):
    nba[season_id] = load_nba(season)
    nba_combined = update_combined(nba_combined, nba[season_id])

# Custom scorer using calculate_score

def custom_score_func(estimator, X, y):

    probs = estimator.predict_proba(X)
    # Prepare top_indices as in main code
    top_indices = {1: [], 2: [], 3: []}
    results_dict = {}
    true_results_dict = {}
    for team in np.unique(y)[1:]:
        team = int(team)
        team_names = {1: "first all-nba team", 2: "second all-nba team", 3: "third all-nba team"}
        team_key = team_names.get(team)
        indices = np.where(y == team)[0]
        players = X.index[indices].tolist()
        true_results_dict[team_key] = players
    for team in np.unique(y)[1:]:
        team = int(team)
        i = 0
        while len(top_indices[team]) < 5:
            top = np.argsort(probs[:, team])[::-1][i]
            if top not in np.concatenate(list(top_indices.values())):
                top_indices[team].append(top)
            i += 1
        team_names = {1: "first all-nba team", 2: "second all-nba team", 3: "third all-nba team"}
        team_key = team_names.get(team)
        # Assume X has a corresponding record attribute (if not, adapt as needed)
        players = X.index[top_indices[team]].tolist()
        results_dict[team_key] = players

    score = calculate_score(results_dict, true_results_dict)
    max_score = calculate_score(true_results_dict, true_results_dict)
    print(f"Score: {score} / {max_score}")
    return score / max_score

# custom_scorer = make_scorer(custom_score_func, greater_is_better=True)

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 95, 110],
    # 'max_features': [2, 3],
    # 'min_samples_leaf': [3, 4, 5],
    # 'min_samples_split': [8, 10, 12],
    # 'n_estimators': [100, 200]
}

custom_score = make_scorer(custom_score_func, needs_proba=True)

skf = StratifiedKFold(n_splits=3)
clf = GridSearchCV(RandomForestClassifier(), param_grid, scoring=custom_score_func, 
                   cv = skf,
                   n_jobs=-1)
                #    cv=((slice(None), slice(None))))
                #    cv=3)
clf.fit(nba_combined.data, nba_combined.target.values.flatten())
nba_predict = load_nba(TEST)
probs = clf.predict_proba(nba_predict.data)
players_team = np.zeros(len(nba_predict.target))
true_team = np.zeros(len(nba_predict.target))
print(f"VALUES OF LAST CLF: {clf.best_estimator_.get_params()}")
for team in nba_combined.target['ALL_NBA_TEAM'].unique()[1:]:
    team = int(team)
    i = 0
    while len(top_indices[team]) < 5 :
        top = np.argsort(probs[:, team])[::-1][i]
        if top not in np.concatenate(top_indices):
            top_indices[team].append(top)
        i+=1
    # Map team index to team name
    team_names = {
        1: "first all-nba team",
        2: "second all-nba team",
        3: "third all-nba team"
    }
    players_team[top_indices[team]] = team
    team_key = team_names.get(team)
    players = nba_predict.record.iloc[top_indices[team]].tolist()
    results_dict[team_key] = players

    indices = nba_predict.target.index[nba_predict.target.astype(int) == team].tolist()
    players = nba_predict.record.loc[indices].tolist()
    # Convert pandas indices to numpy array of integer positions
    indices_np = nba_predict.target.index.get_indexer(indices)
    true_team[indices_np] = team
    true_results_dict[team_key] = players

with open("output/predicted.json", "w") as f:
    json.dump(results_dict, f, indent=2)
with open("output/ground_truth.json", "w") as f:
    json.dump(true_results_dict, f, indent=2)



ground_truth_nba = load_json_file(GROUND_TRUTH_NBA)
predicted = load_json_file(PREDICTED_NBA)
score = calculate_score(predicted, ground_truth_nba)
max_score = calculate_score(ground_truth_nba, ground_truth_nba)
print(f'Your score: {score} / {max_score}!')

experiment_name = 'unknown_TeamPrediction'
if max_score == 270:
    experiment_name = 'all-nba_TeamPrediction'
elif max_score == 180:
    experiment_name = 'all-rookie_TeamPrediction'

if isinstance(clf, RandomForestClassifier):
    model_name = 'brute_model_randomForest'
else:
    model_name = 'different_model'

predicted_pd = pd.DataFrame.from_dict(predicted)
# Ensure both arrays are integers and have matching values for labels
# Align y_true and y_pred by index to ensure correct mapping
# y_true = nba_predict.target[nba_predict.target > 0].astype(int)
# y_pred = players_team[players_team > 0].astype(int)
# y_true = true_team[true_team > 0].astype(int)
cm = confusion_matrix(true_team, players_team, labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
fig, ax = plt.subplots()
disp.plot(ax=ax)
# plt.close(fig)  # Prevents duplicate display in some environments


log_to_mlflow(clf,
                experiment_name,
                model_name,
                params=clf.get_params(),
                metrics={"max_score": max_score,
                        "score": score,
                        "score_perc": score / max_score * 100},
                input_data=nba_predict.data,
                charts={"confusion matrix": fig})


# acc = accuracy_score(nba_predict.target, pred)
# print(acc)


a =  5
