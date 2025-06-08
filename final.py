import json
import mlflow
import mlflow.sklearn
from lib.nba_dataset import load_nba, load_rookies, filter_dataset
from lib.check_score import calculate_score
from lib.metric import get_weighted_team_predictions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

def na_func(col):
    if col.dropna().empty:
        return 0
    return col.median()

mlflow.set_tracking_uri("http://127.0.0.1:8080")

allNBA_model_uri = 'runs:/d94e587a6b0d4d088ed0b4fb106a5139/BayesSearchCV_RandomForestClassifier'
allNBA_model = mlflow.sklearn.load_model(allNBA_model_uri)

nba_players = load_nba('2025')
nba_players = filter_dataset(nba_players, drop_low_chances=True, fill_na=na_func)

expected_features = allNBA_model.feature_names_in_
nba_players.data = nba_players.data[expected_features]


probs = allNBA_model.predict_proba(nba_players.data)
results_dict_allNBA, true_results_dict_allNBA, players_team_allNBA, true_team_allNBA = get_weighted_team_predictions(
    probs, nba_players.target, nba_players.record
)

cm_plot, ax = plt.subplots()
cm = confusion_matrix(true_team_allNBA, players_team_allNBA, labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot(ax=ax)
plt.show()


rookies_model_uri = 'runs:/6b70ca74cb354731bc0d99c4b79b05ca/BayesSearchCV_GradientBoostingClassifier'
allROOKIES_model = mlflow.sklearn.load_model(rookies_model_uri)

rookies_players = load_rookies('2025')
rookies_players = filter_dataset(rookies_players, rookies=True, fill_na=na_func)
rookies_players.data = rookies_players.data.fillna(rookies_players.data.median())

expected_features = allROOKIES_model.feature_names_in_
rookies_players.data = rookies_players.data[expected_features]


probs = allROOKIES_model.predict_proba(rookies_players.data)
results_dict_allROOKIES, true_results_dict_allROOKIES, players_team_allROOKIES, true_team_allROOKIES = get_weighted_team_predictions(
    probs, rookies_players.target, rookies_players.record
)






merged_results = {**results_dict_allNBA, **results_dict_allROOKIES}
merged_true_results = {**true_results_dict_allNBA, **true_results_dict_allROOKIES}

with open("final_output/predicted_combined.json", "w") as f:
    json.dump(merged_results, f, indent=2)
with open("final_output/ground_truth_combined.json", "w") as f:
    json.dump(merged_true_results, f, indent=2)

score = calculate_score(merged_results, merged_true_results)
max_score = calculate_score(merged_true_results, merged_true_results)
print(f'score: {score} / {max_score}')
