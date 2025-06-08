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

def fetch_data():
    urls = ['https://www.basketball-reference.com/awards/all_league.html',
            'https://www.basketball-reference.com/awards/all_rookie.html']

    for url in urls:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find_all('table')
        df = pd.read_html(StringIO(str(table)))[0]
        match = re.search(r'/([^/]+)\.html$', url)
        with open(f'data/{match.group(1)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(df, f)

def get_all_nba_teams(season=None) -> pd.DataFrame:
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
    with open('data/all_league.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        df = pickle.load(f)
    df = df.dropna(axis='index')
    df = df.drop(['Voting', 'Lg'], axis='columns')
    df = df.rename(columns={"Tm": "Team", 'Unnamed: 4': "Player1", 'Unnamed: 5': "Player2", 'Unnamed: 6': "Player3", 'Unnamed: 7': "Player4", 'Unnamed: 8': "Player5"})
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(['1st', '2nd', '3rd'], [1, 2, 3])
    df = df.map(lambda x: x.rsplit(' ', 1)[0] if isinstance(x, str) else x)
    all_league_df = df.set_index(['Season', 'Team'])
    if season != None:
        all_league_df = all_league_df.loc[season].reset_index().melt(id_vars=['Team'], value_vars=['Player1', 'Player2', 'Player3', 'Player4', 'Player5'], 
                                                 var_name='Player_Position', value_name='Player')[['Player', 'Team']]
    return all_league_df

def get_all_rookie_teams(season=None) -> pd.DataFrame:
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
    with open('data/all_rookie.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        df = pickle.load(f)
    df = df.dropna(axis='index')
    df = df.drop(['Voting', 'Lg'], axis='columns')
    df = df.rename(columns={"Tm": "Team", 'Unnamed: 4': "Player1", 'Unnamed: 5': "Player2", 'Unnamed: 6': "Player3", 'Unnamed: 7': "Player4", 'Unnamed: 8': "Player5"})
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(['1st', '2nd'], [1, 2])
    # df = df.map(lambda x: x.rsplit(' ', 1)[0] if isinstance(x, str) else x)
    all_rookie_df = df.set_index(['Season', 'Team'])
    if season != None:
        all_rookie_df = all_rookie_df.loc[season].reset_index().melt(id_vars=['Team'], value_vars=['Player1', 'Player2', 'Player3', 'Player4', 'Player5'], 
                                                 var_name='Player_Position', value_name='Player')[['Player', 'Team']]
    return all_rookie_df

def get_stats(player_full_name : str) -> pd.DataFrame:
    player_id_ = players.find_players_by_full_name(player_full_name)[0]['id']
    stats = playercareerstats.PlayerCareerStats(player_id=player_id_)
    return stats.get_data_frames()[0]

def get_players_stats_season(season : str) -> pd.DataFrame:
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
    stats = stats.get_data_frames()[0]
    stats : pd.DataFrame = stats[['PLAYER_ID', 'PLAYER_NAME', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', \
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', \
            'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3']]
    return stats

def load_nba(SEASON = '2022-23'):
    team = get_all_nba_teams(SEASON)
    player_ids = np.zeros(len(team), dtype=object)
    for id_name, name in enumerate(team.Player.values):
        try:
            player_ids[id_name] = players.find_players_by_full_name(name)[0]['id']
        except Exception:
            player_ids[id_name] = None
    team.insert(0, 'Player ID', player_ids)

    df = get_players_stats_season(SEASON)
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(int)
    nba_team = np.zeros(len(df), dtype=int)
    for id_val, value in enumerate(df.values):
        if value[0] in team['Player ID'].values:
            nba_team[id_val] = team[team['Player ID'].values == value[0]].Team.values[0]
        else:
            nba_team[id_val] = 0

    record = df.set_index('PLAYER_ID')['PLAYER_NAME']
    df = df.drop(['PLAYER_NAME'], axis=1)
    feature_names = df.columns.values
    target_name = 'ALL_NBA_TEAM'

    data_df = pd.DataFrame(df.drop(columns=['PLAYER_ID']).values, columns=df.drop(columns=['PLAYER_ID']).columns, copy=False, index=df.PLAYER_ID)
    target_df = pd.Series(nba_team, name=target_name, index=df.PLAYER_ID, dtype=int)  # Convert numpy array to list
    combined_df = pd.concat([data_df, target_df], axis=1)

    fdescr = """ DATASET OF ALL-NBA TEAMS """

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
    combined.data = pd.concat([combined.data, update.data])
    combined.target = pd.concat([combined.target, update.target])
    combined.record = pd.concat([combined.record, update.record])
    # combined.target = pd.concat([combined.target, update.target])
    return combined

if __name__ == '__main__':
    fetch_data()

    # URL of the Basketball Reference All-Rookie Teams page
    # url = 'https://www.basketball-reference.com/awards/all_rookie.html'
    
    # response = requests.get(url)
    # response.raise_for_status()  # Raise an exception for HTTP errors
    # response.encoding = response.apparent_encoding
    
    # soup = BeautifulSoup(response.text, 'html.parser')
    # table = soup.find_all('table')

    # df = pd.read_html(StringIO(str(table)))[0]

# data_json = all_league_df.to_json(orient='records', lines=True)
# json_file_path = '/home/krasa-35/ws/RiSA/WZUM/Project/NBA_Teams.json'
# with open(json_file_path, 'w') as file:
#     file.write(data_json)  # Write each record followed by a comma

# Read the JSON file back into a DataFrame
# with open(json_file_path, 'r') as file:
#     json_data = file.read()
#     # Remove trailing commas from the JSON data
#     cleaned_json_data = json_data.rstrip(',\n')
#     nba_teams_df = pd.read_json(StringIO(cleaned_json_data))
    # a = 4

# Display the result
# print(tabulate(all_league_df.values, headers=all_league_df.columns))
# print(tabulate(first_team_2023.values, headers=first_team_2023.columns))

# Filter the DataFrame for the 2023 season and First Team
# first_team_2023 = all_league_df[
#     (all_league_df['Season'] == '2023-24') & (all_league_df['Tm'] == '1st')
# ]