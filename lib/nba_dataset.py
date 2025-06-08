from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats
from nba_api.stats.static import players
from sklearn.utils import Bunch

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import pickle
import re
import numpy as np
import os
from datetime import datetime

NBA_TEAM_ABBREVIATIONS = {
    "Boston Celtics": "BOS",
    "New York Knicks": "NYK",
    "Milwaukee Bucks": "MIL",
    "Cleveland Cavaliers": "CLE",
    "Orlando Magic": "ORL",
    "Indiana Pacers": "IND",
    "Philadelphia 76ers": "PHI",
    "Miami Heat": "MIA",
    "Chicago Bulls": "CHI",
    "Atlanta Hawks": "ATL",
    "Brooklyn Nets": "BRK",
    "Toronto Raptors": "TOR",
    "Charlotte Hornets": "CHA",
    "Washington Wizards": "WAS",
    "Detroit Pistons": "DET",
    "Oklahoma City Thunder": "OKC",
    "Denver Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN",
    "Los Angeles Clippers": "LAC",
    "Dallas Mavericks": "DAL",
    "Phoenix Suns": "PHO",
    "New Orleans Pelicans": "NOP",
    "Los Angeles Lakers": "LAL",
    "Sacramento Kings": "SAC",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Utah Jazz": "UTA",
    "Memphis Grizzlies": "MEM",
    "San Antonio Spurs": "SAS",
    "Portland Trail Blazers": "POR",
}

NBA_TEAM_NAMES = {abbr: name for name, abbr in NBA_TEAM_ABBREVIATIONS.items()}

def _load_data(file_path: str, season: str) -> pd.DataFrame:
    season = str(season)
    current_year = datetime.now().year
    if not (season.isdigit() and len(season) == 4 and 1947 <= int(season) <= current_year):
        raise ValueError(f"season must be a 4-digit year string between 1947 and {current_year}, got '{season}'")
    if not os.path.exists(file_path):
        fetch_data(season, Teams=True)
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{lvl0}_{lvl1}" if lvl1 and not lvl0.lower().startswith('unnamed') else lvl1
            for lvl0, lvl1 in zip(df.columns.get_level_values(0), df.columns.get_level_values(-1))
        ]
    return df

def fetch_data(season, Teams=False):
    season = str(season)
    urls = [f'https://www.basketball-reference.com/leagues/NBA_{season}_totals.html',
            f'https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html',
            f'https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html',
            f'https://www.basketball-reference.com/leagues/NBA_{season}_shooting.html',
            f'https://www.basketball-reference.com/leagues/NBA_{season}_standings.html',
            f'https://www.basketball-reference.com/leagues/NBA_{season}_rookies.html']

    if Teams:
        urls.append('https://www.basketball-reference.com/awards/all_league.html')
        urls.append('https://www.basketball-reference.com/awards/all_rookie.html')

    for url in urls:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find_all('table')
        # If there are 6 tables, choose the 5th one; otherwise, use the first table
        match = re.search(r'/([^/]+)\.html', url)
        if url == urls[4]:
            for i in [0, 1]:
                df = pd.read_html(StringIO(str(table[i])))[0]
                with open(f'data/{match.group(1)}_{i}.pkl', 'wb') as f:
                    pickle.dump(df, f)
        else:
            df = pd.read_html(StringIO(str(table[0])))[0]
            with open(f'data/{match.group(1)}.pkl', 'wb') as f:
                pickle.dump(df, f)

def _get_standing_teams(season: str):
    df0 = _load_data(f'data/NBA_{season}_standings_0.pkl', season)
    df1 = _load_data(f'data/NBA_{season}_standings_1.pkl', season)
    # Remove the first row if it's a header row (sometimes happens with pd.read_html)
    if isinstance(df0.iloc[0, 1], str) and df0.iloc[0, 1] == 'W':
        df0 = df0.iloc[1:].reset_index(drop=True)
    if isinstance(df1.iloc[0, 1], str) and df1.iloc[0, 1] == 'W':
        df1 = df1.iloc[1:].reset_index(drop=True)
    # Rename first column to 'Team' for both
    df0 = df0.rename(columns={df0.columns[0]: "Team"})
    df1 = df1.rename(columns={df1.columns[0]: "Team"})
    # Keep only the first 8 columns (team name + 7 stats)
    df0 = df0.iloc[:, :8]
    df1 = df1.iloc[:, :8]
    # Remove possible summary/empty rows
    df0 = df0[df0['Team'].notna() & (df0['Team'] != '')]
    df1 = df1[df1['Team'].notna() & (df1['Team'] != '')]
    # Remove '*' from end of team names if present
    df0['Team'] = df0['Team'].str.rstrip('*')
    df1['Team'] = df1['Team'].str.rstrip('*')
    # Replace team names with abbreviations
    df0['Team'] = df0['Team'].replace(NBA_TEAM_ABBREVIATIONS)
    df1['Team'] = df1['Team'].replace(NBA_TEAM_ABBREVIATIONS)
    return pd.concat([df0, df1], axis=0, ignore_index=True)

def _get_all_nba_players(season : str) -> pd.DataFrame:
    """
    Fetch and process NBA All-League Teams data from Basketball Reference.

    This function retrieves the NBA All-League Teams data from the Basketball Reference website,
    parses the HTML content, and processes it into a pandas DataFrame. The resulting DataFrame
    contains information about the teams and players for each season, with unnecessary columns
    removed and the data indexed by 'Season' and 'Team'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the NBA All-League Teams data with the following columns:
                      The DataFrame is indexed by 'Season' and 'Team'.

    Example:
        To retrieve the players of Team 1 from the 2023-24 season:
        ```python
        df = get_all_nba_teams()
        team_1_players = df.loc['2023-24', 'Team1']
        print(team_1_players)
        ```
    """
    df = _load_data('data/all_league.pkl', season)
    df = df.dropna(axis='index')
    df = df.drop(['Voting', 'Lg'], axis='columns')
    df = df.rename(columns={"Tm": "Team", 'Unnamed: 4': "Player1", 'Unnamed: 5': "Player2", 'Unnamed: 6': "Player3", 'Unnamed: 7': "Player4", 'Unnamed: 8': "Player5"})
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(['1st', '2nd', '3rd'], [1, 2, 3])
    df = df.map(lambda x: x.rsplit(' ', 1)[0] if isinstance(x, str) else x)
    df['Season'] = df['Season'].apply(lambda s: str(int(s[:4]) + 1) if isinstance(s, str) and '-' in s else s)
    all_league_df = df.set_index(['Season', 'Team'])
    if season is not None:
        all_league_df = all_league_df.loc[season].reset_index().melt(
            id_vars=['Team'],
            value_vars=['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
            var_name='Player_Position',
            value_name='Player'
        )[["Player", "Team"]]
    return all_league_df

def _get_all_rookie_players(season : str) -> pd.DataFrame:
    """
    Fetch and process NBA All-Rookie Teams data from Basketball Reference.

    This function retrieves the NBA All-Rookie Teams data from the Basketball Reference website,
    parses the HTML content, and processes it into a pandas DataFrame. The resulting DataFrame
    contains information about the rookie teams and players for each season, with unnecessary columns
    removed and the data indexed by 'Season' and 'Team'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the NBA All-Rookie Teams data with the following columns:
                      The DataFrame is indexed by 'Season' and 'Team'.

    Example:
        To retrieve the players of Team 1 from the 2023-24 season:
        ```python
        df = get_all_rookie_teams()
        team_1_players = df.loc['2023-24', 'Team1']
        print(team_1_players)
        ```
    """

    df = _load_data('data/all_rookie.pkl', season)
    df = df.dropna(axis='index')
    df = df.drop(['Voting', 'Lg'], axis='columns')
    df = df.rename(columns={"Tm": "Team", 'Unnamed: 4': "Player1", 'Unnamed: 5': "Player2", 'Unnamed: 6': "Player3", 'Unnamed: 7': "Player4", 'Unnamed: 8': "Player5"})
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(['1st', '2nd'], [1, 2])
    # df = df.map(lambda x: x.rsplit(' ', 1)[0] if isinstance(x, str) else x)
    df['Season'] = df['Season'].apply(lambda s: str(int(s[:4]) + 1) if isinstance(s, str) and '-' in s else s)
    all_rookie_df = df.set_index(['Season', 'Team'])
    if season is not None:
        all_league_df = all_rookie_df.loc[season].reset_index().melt(
            id_vars=['Team'],
            value_vars=['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
            var_name='Player_Position',
            value_name='Player'
        )[["Player", "Team"]]
    return all_league_df

def _get_players_stats_season_bs4(season : str) -> pd.DataFrame:
    df_totals   = _load_data(f'data/NBA_{season}_totals.pkl', season)
    df_advanced = _load_data(f'data/NBA_{season}_advanced.pkl', season)
    df_per_game = _load_data(f'data/NBA_{season}_per_game.pkl', season)
    df_shooting = _load_data(f'data/NBA_{season}_shooting.pkl', season)
    teams = _get_standing_teams(season)
    # Clean and prepare DataFrames
    for d in [df_totals, df_advanced, df_per_game, df_shooting]:
        d.drop(d[d['Player'] == 'League Average'].index, inplace=True)
        d.sort_values(['Player', 'Age'], inplace=True)
        d.reset_index(drop=True, inplace=True)

    # Merge all DataFrames on Player and Team, handling common columns with different values
    def merge_dfs(left, right, left_name, right_name):
        # Find common columns (excluding keys)
        common = [col for col in left.columns if col in right.columns and col not in ['Player', 'Team']]
        # For each common column, check if values differ; if so, add suffixes
        for col in common:
            if not left[col].equals(right[col]):
                left.rename(columns={col: f"{col}_{left_name}"}, inplace=True)
                right.rename(columns={col: f"{col}_{right_name}"}, inplace=True)
        # Drop duplicate columns from right (except keys)
        right_to_merge = right[[c for c in right.columns if c not in left.columns or c in ['Player', 'Team']]]
        return pd.merge(left, right_to_merge, on=['Player', 'Team'], how='inner')

    df = merge_dfs(df_totals, df_advanced, 'totals', 'advanced')
    df = merge_dfs(df, df_per_game, 'totadv', 'per_game')
    df = merge_dfs(df, df_shooting, 'totadvpg', 'shooting')

    multi_team_players = df[df['Team'].str.endswith('TM', na=False)]['Player'].unique()

    last_team_map = (
        df[df['Player'].isin(multi_team_players) & (~df['Team'].str.endswith('TM', na=False))]
        .groupby('Player')
        .last()['Team']
        .to_dict()
    )

    def pick_row(player_rows):
        if player_rows['Team'].str.endswith('TM', na=False).any():
            row = player_rows[player_rows['Team'].str.endswith('TM', na=False)].iloc[0].copy()
            if row['Player'] in last_team_map:
                row['Team'] = last_team_map[row['Player']]
            return row
        else:
            return player_rows.iloc[0]

    df = df.groupby('Player', group_keys=False).apply(pick_row).reset_index(drop=True)

    for id_x, player in df.iterrows():
        player_team = player['Team']
        for col in list(teams.columns):
            if f'{col}_team' not in df.columns:
                df[f'{col}_team'] = object()
            # Find the row in teams where 'Team' matches player's team
            if 'Team' in teams.columns and player_team in teams['Team'].values:
                value = teams.loc[teams['Team'] == player_team, col].values
                if len(value) > 0:
                    # Assign value directly without type casting to avoid ValueError for strings
                    df.at[id_x, f'{col}_team'] = value[0]

    return df

def _group_standard_rookies(season: str, df : pd.DataFrame):
    df_rookies = _load_data(f'data/NBA_{season}_rookies.pkl', season)
    rookie_players = set(df_rookies['Player'])
    df_standard = df[~df['Player'].isin(rookie_players)].copy()
    df_rookie = df[df['Player'].isin(rookie_players)].copy()
    return df_standard, df_rookie

def load_nba(SEASON = '2023'):
    all_nba_df = _get_all_nba_players(SEASON)
    df = _get_players_stats_season_bs4(SEASON)
    df, _ = _group_standard_rookies(SEASON, df)

    nba_team = np.zeros(len(df), dtype=object)
    for i, (_, row) in enumerate(df.iterrows()):
        if row['Player'] in all_nba_df['Player'].values:
            nba_team[i] = int(all_nba_df[all_nba_df['Player'] == row['Player']].Team.values[0])
        else:
            nba_team[i] = int(0)

    # df = df.sort_values('Rk_totals')
    # df = df.set_index('Rk_totals')
    df = df.reset_index(drop=True)
    record = df['Player']
    df = df.drop(['Player'], axis=1)
    feature_names = df.columns.values
    target_name = 'ALL_TEAM'

    # Only keep columns that can be converted to float
    numeric_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue

    data_df = pd.DataFrame(df[numeric_cols].values, columns=numeric_cols, copy=False)
    target_df = pd.Series(nba_team, name=target_name, index = data_df.index)
    combined_df = pd.concat([data_df, target_df], axis=1)

    fdescr = """ DATASET OF STANDARD PLAYERS IN NBA """

    return Bunch(
        data=data_df,
        target=target_df,
        record=record,
        frame=combined_df,
        target_names=target_name,
        DESCR=fdescr,
        feature_names=feature_names,
        data_module="sklearn.datasets.data",
    )

def load_rookies(SEASON = '2023'):
    all_rookie_df = _get_all_rookie_players(SEASON)
    df = _get_players_stats_season_bs4(SEASON)
    _, df = _group_standard_rookies(SEASON, df)

    player_name_replacements = {
        "GG Jackson II": "Gregory Jackson",
        "Dereck Lively II": "Dereck Lively",
        "Jaime Jaquez Jr.": "Jaime Jaquez"
    }
    df['Player'] = df['Player'].replace(player_name_replacements)

    rookie_team = np.zeros(len(df), dtype=object)
    for i, (_, row) in enumerate(df.iterrows()):
        if row['Player'] in all_rookie_df['Player'].values:
            rookie_team[i] = int(all_rookie_df[all_rookie_df['Player'] == row['Player']].Team.values[0])
        else:
            rookie_team[i] = int(0)

    # df = df.sort_values('Rk_totals')
    # df = df.set_index('Rk_totals')
    df = df.reset_index(drop=True)
    record = df['Player']
    df = df.drop(['Player'], axis=1)
    feature_names = df.columns.values
    target_name = 'ALL_TEAM'

    # Only keep columns that can be converted to float
    numeric_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue

    data_df = pd.DataFrame(df[numeric_cols].values, columns=numeric_cols, copy=False)
    target_df = pd.Series(rookie_team, name=target_name, index=data_df.index)
    combined_df = pd.concat([data_df, target_df], axis=1)

    fdescr = """ DATASET OF ROOKIES IN NBA """

    return Bunch(
        data=data_df,
        target=target_df,
        record=record,
        frame=combined_df,
        target_names=target_name,
        DESCR=fdescr,
        feature_names=feature_names,
        data_module="sklearn.datasets.data",
    )

def update_combined(combined: Bunch, update: Bunch):
    combined.data = pd.concat([combined.data, update.data], ignore_index=True)
    combined.target = pd.concat([combined.target, update.target], ignore_index=True)
    combined.record = pd.concat([combined.record, update.record], ignore_index=True)
    combined.frame = pd.concat([combined.frame, update.frame], ignore_index=True)
    combined.target_names = combined.target_names
    combined.DESCR = combined.DESCR
    combined.feature_names = combined.feature_names
    combined.data_module="sklearn.datasets.data"
    return combined

def filter_dataset(df : pd.DataFrame, rookies=False, fill_na=None, debug=False, old_ver=False, drop_low_chances=False):
    if debug:
        print(f'len(df.data): {len(df.data)}')
        print(f'len(df.data.columns): {len(df.data.columns)}')
        print("Rows with NA values in df:")
        print(df.data[df.data.isna().any(axis=1)])

    # df.data = df.data.reset_index(drop=True)
    # df.target = df.target.reset_index(drop=True)
    if rookies:
        cols_to_drop = ['Awards', 'MP_totadv', 'FG_totadv',       
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
    else:
        if old_ver:
            cols_to_drop = ['MP_totadv', 'Rk_advanced',
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
        else:
            cols_to_drop = ['Awards', 'MP_totadv', 'FG_totadv', 'FGA_totadv', 'PTS_totadv', '2P_totadv', '2PA_totadv', 'FT_totadv', 'FTA_totadv',
                            'ORB_totadv', 'DRB_totadv', 'TRB_totadv', 'TOV_totadv', 'AST_totadv', 'BLK_totadv', 'STL_totadv',
                            'FG_per_game', 'FGA_per_game', '2P_per_game', '2PA_per_game', '3P_per_game', '3PA_per_game',
                            'FT_per_game', 'FTA_per_game', 'Rk_advanced', 'Rk_totadvpg', 'Rk_shooting',
                            'FG% by Distance_3P', 'FG% by Distance_2P', '% of FGA by Distance_2P', '% of FGA by Distance_3P',
                            '% of FGA by Distance_0-3', '% of FGA by Distance_3-10', '% of FGA by Distance_10-16',
                            '% of FGA by Distance_16-3P', '% of FG Ast\'d_2P', '% of FG Ast\'d_3P', 'Corner 3s_3P%',
                            'Corner 3s_%3PA', 'FG% by Distance_0-3', 'FG% by Distance_3-10', 'FG% by Distance_10-16',
                            'FG% by Distance_16-3P', 'Dunks_%FGA', 'OBPM', 'WS', 'TRB%', 'W_team', 'L_team', 'W/L%_team', '2P%']
            if drop_low_chances:
                mask_low_chances = (
                                    ((df.data['GS'] / df.data['G']) < 0.9) |
                                    (df.data['BPM'] < -0.5) |
                                    (df.data['VORP'] < 1) |
                                    (df.data['PER'] < 10) |
                                    (df.data['G'] < 50)
                                    )            
                mask_low_chances.index = df.data.index
                df.data = df.data[~mask_low_chances].reset_index(drop=True)
                df.target = df.target[~mask_low_chances].reset_index(drop=True)
                df.frame = df.frame[~mask_low_chances].reset_index(drop=True)         
                df.record = df.record[~mask_low_chances].reset_index(drop=True)         
    df.data = df.data.drop(columns=cols_to_drop,errors='ignore')
    df.frame = df.frame.drop(columns=cols_to_drop,errors='ignore')
    df.feature_names = np.setdiff1d(df.feature_names, cols_to_drop)
    if debug:
        print("-----------AFTER FILTRATION---------")
        print(f'len(df.data): {len(df.data)}')
        print(f'len(df.data.columns): {len(df.data.columns)}')
        train_nans = df.data.isna().sum()
        train_nans = train_nans[train_nans > 0]
        if not train_nans.empty:
            print("\nColumns with most NaNs in training data:")
            print(train_nans.sort_values(ascending=False))
            
    if fill_na is not None:
        if debug:
            print(f"Found {df.data.isna().sum().sum()} values with NaN, they will be completed.")
        if callable(fill_na):
            df.data = df.data.apply(lambda x: x.fillna(fill_na(x)), axis=0)
        else:
            raise ValueError("fill_na must be a callable.")

    return df

if __name__ == '__main__':
    # fetch_data(season='2020', Teams=True)
    # fetch_data()
    # fetch_data('2025')
    # df = get_players_stats_season_bs4('2025')
    # for i in range(2000, 2025, 1):
    #     fetch_data(str(i),Teams=False)
    df_standard = load_nba('2001')
    df_rookies = load_rookies('2001')
    a = 5
