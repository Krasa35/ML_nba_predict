from nba_api.stats.endpoints import playerawards
from nba_api.stats.static import players
import pandas as pd
from tabulate import tabulate
import time

def is_all_nba_team(player_awards: pd.DataFrame, season_nr='2023-24'):
    """
    Check if a player was selected to any All-NBA team in a given season and return the team number.

    Args:
        player_id (int): The unique identifier for the player.
        season_nr (str): The NBA season in question (e.g., '2023-24'). Defaults to '2023-24'.

    Returns:
        int: The All-NBA team number (1 for First Team, 2 for Second Team, etc.) if the player was selected, 
             or 0 if the player was not selected to any All-NBA team.
    """
    player_awards_data = player_awards.query(f'DESCRIPTION == \'All-NBA\' and SEASON == \'{season_nr}\'')
    if not player_awards_data.empty:
        return int(player_awards_data['ALL_NBA_TEAM_NUMBER'].iloc[0])
    return 0

seasons = [str(20)+str(f"{n:02d}")+'-'+str(f"{(n+1):02d}") for n in range(25)]
[203999, 2544]
#Jokic, lebron


players_pd = pd.DataFrame(data=players.players, columns=players.get_players()[0].keys())
# print(tabulate(players_pd.head(), headers=players_pd.columns, tablefmt="rounded_grid"))
# player_awards = playerawards.PlayerAwards(player_id=players_pd.id[:1000].values, timeout=100).get_data_frames()
# player_awards = playerawards.PlayerAwards(player_id=[203999, 2544], timeout=100).get_data_frames()
# print(all_star_team1_season_25)
# print(tabulate(all_star_team1_season_25[0], tablefmt="rounded_grid"))
# LeBron James
# awards = playerawards.PlayerAwards(player_id='2544')
# awards_pd = pd.DataFrame([awards.get_data_frames])
# awards_pd = awards.get_data_frames()
# awards_filtered_pd : pd.DataFrame = awards_pd[0].drop(['PERSON_ID', 'TYPE', 'CONFERENCE', 'SUBTYPE1', 'SUBTYPE2', 'SUBTYPE3', 'MONTH', 'WEEK'], axis=1)
# awards_filtered_pd = awards_filtered_pd[awards_filtered_pd['DESCRIPTION'] == 'All-NBA']
players = 4172
# nba_team = [[],[],[]]
for player_id, player_name in zip(players_pd.id[players:], players_pd.last_name[players:]):
    player_awards: pd.DataFrame = playerawards.PlayerAwards(player_id=player_id).get_data_frames()[0]
    time.sleep(3)
    players += 1
    if not player_awards.empty:
        player_awards_data = player_awards.query('ALL_NBA_TEAM_NUMBER.notnull() and ALL_NBA_TEAM_NUMBER != \"\"')
        player_awards_json = player_awards_data.to_json(orient='records')
        if not player_awards_data.empty:
            with open('/home/krasa-35/ws/RiSA/WZUM/Project/player_awards.json', 'a') as file:
                file.write(player_awards_json + ',\n')  # Write each record followed by a comma
        # for season in seasons:
            # team = is_all_nba_team(player_awards, season) - 1
            # if team >= 0:
                # nba_team[team].append(player_name)

# print(tabulate(awards_filtered_pd, headers=awards_filtered_pd.columns, tablefmt="rounded_grid"))
player_id = 2544
print(f"Player ID: {player_id}, is_allNBATeam(player_id, team_nr=3): {is_all_nba_team(player_id)}")
# awards_dict = awards.get_normalized_dict()
# print(awards_pd[0])
# print([awards.get])
a = 5
# print(awards_pd.head())