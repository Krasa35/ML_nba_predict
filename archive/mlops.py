from nba_api.stats.endpoints import playercareerstats

# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id=['203999', '2544']) 

# pandas data frames (optional: pip install pandas)
carrer_df = career.get_data_frames()[1]

# json
career.get_json()

# dictionary
career.get_dict()
a = 5