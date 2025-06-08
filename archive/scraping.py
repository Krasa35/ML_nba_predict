import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from io import StringIO

# URL of the Basketball Reference All-League Teams page
url = 'https://www.basketball-reference.com/awards/all_league.html'

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Raise an exception for HTTP errors

# Ensure the correct encoding is set
response.encoding = response.apparent_encoding

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables on the page
table = soup.find_all('table')

# Iterate through all tables and convert them to DataFrames
df = pd.read_html(str(table))[0]
all_league_df = df.dropna(axis='index')

# Save the DataFrame to a JSON file
data_json = all_league_df.to_json(orient='records', lines=True)
json_file_path = '/home/krasa-35/ws/RiSA/WZUM/Project/NBA_Teams.json'
with open(json_file_path, 'w') as file:
    file.write(data_json)  # Write each record followed by a comma

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
first_team_2023 = all_league_df[
    (all_league_df['Season'] == '2023-24') & (all_league_df['Tm'] == '1st')
]