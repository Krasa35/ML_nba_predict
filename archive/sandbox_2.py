from lib.nba_dataset import load_nba, update_combined

from sklearn.utils import Bunch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


START = 2000
LEARN = 25
# TEST  = '2023'
GROUND_TRUTH_NBA = "output/ground_truth.json"
PREDICTED_NBA = "output/predicted.json"
# seasons = [f"{START + n}-{str(START + n + 1 - int(START / 100) * 100).zfill(2)}" for n in range(LEARN)]
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
# Update the loop
for season_id, season in enumerate(seasons):
    nba[season_id] = load_nba(season)
    nba_combined = update_combined(nba_combined, nba[season_id])

# Filter out rows where target == 0
nba_combined.data = nba_combined.data.reset_index(drop=True)
nba_combined.target = nba_combined.target.reset_index(drop=True)
data = nba_combined.data.reset_index(drop=True)
target = nba_combined.target.reset_index(drop=True)
mask = nba_combined.target['ALL_TEAM'] != 0
filtered_data = nba_combined.data.loc[mask]
filtered_target = nba_combined.target.loc[mask]

# Combine filtered data and target into a DataFrame
df = filtered_data.copy()
df['target'] = np.ravel(filtered_target.reset_index(drop=True).values)

# Ensure all columns are numeric and drop columns with all NaN values
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)
if 'target' in corr_matrix.columns:
    target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
else:
    raise ValueError("'target' not found in correlation matrix columns: {}".format(corr_matrix.columns))

# # Plot top 10 positively correlated features
# top_pos = target_corr.head(10)
# if not top_pos.empty:
#     plt.figure(figsize=(12, 6))
#     top_pos.plot(kind='bar', color='green', label='Top Positive')
#     plt.title('Top 10 Features Positively Correlated with Target')
#     plt.ylabel('Correlation')
#     plt.xlabel('Feature')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No positively correlated features to plot.")

# # Plot top 10 negatively correlated features
# top_neg = target_corr.tail(10)
# if not top_neg.empty:
#     plt.figure(figsize=(12, 6))
#     top_neg.plot(kind='bar', color='red', label='Top Negative')
#     plt.title('Top 10 Features Negatively Correlated with Target')
#     plt.ylabel('Correlation')
#     plt.xlabel('Feature')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No negatively correlated features to plot.")

# # Scatter plots for top 3 absolute correlated features
# for feature in target_corr.abs().sort_values(ascending=False).head(5).index:
#     plt.figure(figsize=(8, 5))
#     plt.scatter(df[feature], df['target'], alpha=0.5)
#     plt.title(f'Scatter Plot: {feature} vs Target')
#     plt.xlabel(feature)
#     plt.ylabel('Target')
#     plt.tight_layout()
#     plt.show()

# Print features with correlation > 0.9 or < -0.9 (excluding 'target' itself)
# Print pairs of features with correlation magnitude > 0.9 (excluding self-correlation)
threshold = 0.9
corr_pairs = corr_matrix.abs().unstack()
# Remove self-correlations
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
# Keep only one of each pair
corr_pairs = corr_pairs.drop_duplicates()
high_corr_pairs = corr_pairs[corr_pairs > threshold]

i = 0
n=0
for g, gs in zip(df.G, df.GS):
    if gs < g-1:
        print(f'g: {g}, gs: {gs}')
    else:
        i+=1
    n+=1
print(f'same values in {i}/{n} times.')

if not high_corr_pairs.empty:
    print(f"Feature pairs with correlation magnitude > {threshold}")
    print(f"Found {len(high_corr_pairs)} highly correlated pairs:")
    for (feat1, feat2), corr_val in high_corr_pairs.items():
        print(f"{feat1} & {feat2}: {corr_val:.3f}")
else:
    print(f"No feature pairs with correlation magnitude > {threshold}.")