import numpy as np
import pandas as pd
from lib.check_score import calculate_score
import matplotlib.pyplot as plt


def get_weighted_team_predictions(probs, target, record):
    # if isinstance(record, (pd.Index, np.ndarray)):
    #     record = pd.Series(record, index=record)
    # if not target.index.is_unique:
    #     target = target.reset_index(drop=True)
    #     record = record.reset_index(drop=True)
    top_indices = {1: [], 2: [], 3: []}
    results_dict = {}
    true_results_dict = {}
    players_team = np.zeros(len(target))
    true_team = np.zeros(len(target))    
    teams = sorted(target.unique())[1:]
    if len(teams) == 3:
        team_names = {
            1: "first all-nba team",
            2: "second all-nba team",
            3: "third all-nba team"
        }
    else:
        team_names = {
            1: "first rookie all-nba team",
            2: "second rookie all-nba team",
        }
    for team in teams:
        team = int(team)
        i = 0
        while len(top_indices[team]) < 5:
            weighted_probs = 2 * probs[:, team] + sum([
                probs[:, t] for t in teams
            ])
            top = np.argsort(weighted_probs)[::-1][i]
            if top not in np.concatenate(list(top_indices.values())):
                top_indices[team].append(top)
            i += 1
        players_team[top_indices[team]] = team
        team_key = team_names.get(team)
        players = record.iloc[top_indices[team]].values.flatten().tolist()
        results_dict[team_key] = players

        indices = target.index[target.astype(int) == team].tolist()
        players = record.iloc[indices].values.flatten().tolist()
        indices_np = target.index.get_indexer(indices)
        true_team[indices_np] = team
        true_results_dict[team_key] = players
    return results_dict, true_results_dict, players_team, true_team

def custom_score_func(estimator, X, y, record):
    probs = estimator.predict_proba(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    results_dict, true_results_dict, _, _ = get_weighted_team_predictions(probs, y, record)
    score = calculate_score(results_dict, true_results_dict)
    max_score = calculate_score(true_results_dict, true_results_dict)
    print(f"Score: {score} / {max_score}")
    return score / max_score

def get_corr_matrix(bunch, only_super_team=False, return_whole = False) -> pd.DataFrame:
    data = bunch.data.reset_index(drop=True)
    target = bunch.target.reset_index(drop=True)
    if only_super_team:
        mask = target != 0
        data = data.loc[mask.values]
        target = target.loc[mask.values]


    # Combine filtered data and target into a DataFrame
    df = data.copy()
    df['target'] = np.ravel(target.reset_index(drop=True).values)

    # Ensure all columns are numeric and drop columns with all NaN values
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')

    # Compute correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    if return_whole:
        return corr_matrix
    else:
        if 'target' in corr_matrix.columns:
            target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
        else:
            raise ValueError("'target' not found in correlation matrix columns: {}".format(corr_matrix.columns))
        return target_corr

def get_corr_based_plot(corr_matrix : pd.DataFrame, plot_type = 0, df = None):
    top_pos = corr_matrix.head(10)
    if not top_pos.empty and plot_type==10:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_pos.plot(kind='bar', color='green', label='Top Positive', ax=ax)
        ax.set_title('Top 10 Features Positively Correlated with Target')
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Feature')
        fig.tight_layout()
        return fig

    top_neg = corr_matrix.tail(10)
    if not top_neg.empty and plot_type==20:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_neg.plot(kind='bar', color='red', label='Top Negative', ax=ax)
        ax.set_title('Top 10 Features Negatively Correlated with Target')
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Feature')
        fig.tight_layout()
        return fig

    # Scatter plots for top 5 absolute correlated features with target
    if int(plot_type / 10) == 3:
        # Determine which feature to plot based on plot_type % 10
        feature_idx = plot_type % 10
        sorted_features = corr_matrix.abs().sort_values(ascending=False)
        if feature_idx < len(sorted_features):
            feature = sorted_features.index[feature_idx]
        else:
            feature = sorted_features.index[0]  # fallback to the first if out of range

        if df is None:
            raise ValueError("Original DataFrame required for scatter plot. Attach as 'df'.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df.data[feature], df.target, alpha=0.5)
        ax.set_title(f'Scatter Plot: {feature} vs Target')
        ax.set_xlabel(feature)
        ax.set_ylabel('Target')
        fig.tight_layout()
        return fig

def get_corr_charts(nba_combined):
    corr = get_corr_matrix(nba_combined, only_super_team=True)
    return {
        "correlation plot": get_corr_based_plot(corr, plot_type=10),
        "anti correlation plot": get_corr_based_plot(corr, plot_type=20),
        "feature 1 vs target corr": get_corr_based_plot(corr, plot_type=30, df=nba_combined),
        "feature 2 vs target corr": get_corr_based_plot(corr, plot_type=31, df=nba_combined),
        "feature 3 vs target corr": get_corr_based_plot(corr, plot_type=32, df=nba_combined),
    }